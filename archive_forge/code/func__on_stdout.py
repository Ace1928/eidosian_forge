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