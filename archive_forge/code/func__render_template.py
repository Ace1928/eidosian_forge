from __future__ import annotations
import ast
import html
import json
import logging
import os
import pathlib
import re
import sys
import traceback
import urllib.parse as urlparse
from contextlib import contextmanager
from types import ModuleType
from typing import IO, Any, Callable
import bokeh.command.util
from bokeh.application.handlers.code import CodeHandler
from bokeh.application.handlers.code_runner import CodeRunner
from bokeh.application.handlers.handler import Handler, handle_exception
from bokeh.core.types import PathLike
from bokeh.document import Document
from bokeh.io.doc import curdoc, patch_curdoc, set_curdoc as bk_set_curdoc
from bokeh.util.dependencies import import_required
from ..config import config
from .mime_render import MIME_RENDERERS
from .profile import profile_ctx
from .reload import record_modules
from .state import state
def _render_template(self, doc, path):
    """Renders template containing cell outputs.

        Creates an EditableTemplate containing all cell outputs
        found in the notebook and lays them out according to the
        cell metadata (if present).

        Arguments
        ----------
        doc (Document)
            A ``Document`` to render the template into
        path (str):
            The path to the application code.
        """
    from ..config import config
    from ..layout import Column
    from ..pane import panel
    from .state import state
    config.template = 'editable'
    persist = state._jupyter_kernel_context
    editable = 'editable' in state.session_args
    reset = 'reset' in state.session_args
    if not (editable or persist):
        state.template.editable = False
    state.template.title = os.path.splitext(os.path.basename(path))[0].replace('_', ' ').title()
    layouts, outputs, cells = ({}, {}, {})
    for cell_id, objects in state._cell_outputs.items():
        if reset:
            spec = {}
        elif cell_id in self._layout.get('cells', {}):
            spec = self._layout['cells'][cell_id]
        else:
            spec = state._cell_layouts[self].get(cell_id, {})
        panels = [panel(obj) for obj in objects if obj is not None]
        pout = Column(*panels, **self._compute_layout(spec, panels))
        for po in pout:
            po.sizing_mode = pout.sizing_mode
        outputs[cell_id] = pout
        layouts[id(pout)] = spec
        cells[cell_id] = id(pout)
        pout.servable()
    import nbformat
    nb = nbformat.read(self._runner._path, nbformat.NO_CONVERT)
    if 'order' in self._layout:
        cell_order = self._layout['order']
    else:
        cell_order = nb['metadata'].get('panel-cell-order', [])
    ordered = {}
    for cell_id in cell_order:
        if cell_id not in cells or reset:
            continue
        obj_id = cells[cell_id]
        ordered[obj_id] = layouts[obj_id]
        for cell_id in self._layout.get('order', []):
            if cell_id not in cells:
                continue
            obj_id = cells[cell_id]
            ordered[obj_id] = layouts[obj_id]
        for obj_id, spec in layouts.items():
            if obj_id not in ordered:
                ordered[obj_id] = spec
    state.template.param.update(layout=ordered, local_save=not bool(state._jupyter_kernel_context))
    if reset:

        def unset_reset():
            query = state.location.query_params
            query.pop('reset', None)
            search = urlparse.urlencode(query)
            state.location.search = f'?{search}' if search else ''
        state.onload(unset_reset)
    if persist:
        state.template.param.watch(self._update_position_metadata, 'layout')
    state._session_outputs[doc] = outputs