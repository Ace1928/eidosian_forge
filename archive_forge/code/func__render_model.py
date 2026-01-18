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
def _render_model(self, doc: Optional[Document]=None, comm: Optional[Comm]=None) -> 'Model':
    if doc is None:
        doc = Document()
    if comm is None:
        comm = state._comm_manager.get_server_comm()
    model = self.get_root(doc, comm)
    if self._design and self._design.theme.bokeh_theme:
        doc.theme = self._design.theme.bokeh_theme
    if config.embed:
        embed_state(self, model, doc, json=config.embed_json, json_prefix=config.embed_json_prefix, save_path=config.embed_save_path, load_path=config.embed_load_path, progress=False)
    else:
        add_to_doc(model, doc)
    return model