from __future__ import annotations
import asyncio
import functools
import logging
import os
import sys
import threading
import traceback
import uuid
from typing import (
import param  # type: ignore
from bokeh.core.serialization import DeserializationError
from bokeh.document import Document
from bokeh.resources import Resources
from jinja2 import Template
from pyviz_comms import Comm  # type: ignore
from ._param import Align, Aspect, Margin
from .config import config, panel_extension
from .io import serve
from .io.document import create_doc_if_none_exists, init_doc
from .io.embed import embed_state
from .io.loading import start_loading_spinner, stop_loading_spinner
from .io.model import add_to_doc, patch_cds_msg
from .io.notebook import (
from .io.save import save
from .io.state import curdoc_locked, set_curdoc, state
from .util import escape, param_reprs
from .util.parameters import get_params_to_inherit
class MimeRenderMixin:
    """
    Mixin class to allow rendering and syncing objects in notebook
    contexts.
    """

    def _on_msg(self, ref: str, manager, msg) -> None:
        """
        Handles Protocol messages arriving from the client comm.
        """
        root, doc, comm = state._views[ref][1:]
        patch_cds_msg(root, msg)
        held = doc.callbacks.hold_value
        patch = manager.assemble(msg)
        doc.hold()
        try:
            patch.apply_to_document(doc, comm.id if comm else None)
        except DeserializationError:
            self.param.warning('Comm received message that could not be deserialized.')
        finally:
            doc.unhold()
            if held:
                doc.hold(held)

    def _on_error(self, ref: str, error: Exception) -> None:
        if ref not in state._handles or config.console_output in [None, 'disable']:
            return
        handle, accumulator = state._handles[ref]
        formatted = '\n<pre>' + escape(traceback.format_exc()) + '</pre>\n'
        if config.console_output == 'accumulate':
            accumulator.append(formatted)
        elif config.console_output == 'replace':
            accumulator[:] = [formatted]
        if accumulator:
            handle.update({'text/html': '\n'.join(accumulator)}, raw=True)

    def _on_stdout(self, ref: str, stdout: Any) -> None:
        if ref not in state._handles or config.console_output in [None, 'disable']:
            return
        handle, accumulator = state._handles[ref]
        formatted = ['%s</br>' % o for o in stdout]
        if config.console_output == 'accumulate':
            accumulator.extend(formatted)
        elif config.console_output == 'replace':
            accumulator[:] = formatted
        if accumulator:
            handle.update({'text/html': '\n'.join(accumulator)}, raw=True)

    def _render_mimebundle(self, model: Model, doc: Document, comm: Comm, location: Location | None=None):
        from .models.comm_manager import CommManager
        ref = model.ref['id']
        manager = CommManager(comm_id=comm.id, plot_id=ref)
        client_comm = state._comm_manager.get_client_comm(on_msg=functools.partial(self._on_msg, ref, manager), on_error=functools.partial(self._on_error, ref), on_stdout=functools.partial(self._on_stdout, ref), on_open=lambda _: comm.init())
        self._comms[ref] = (comm, client_comm)
        manager.client_comm_id = client_comm.id
        return render_mimebundle(model, doc, comm, manager, location)