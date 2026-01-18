from __future__ import annotations
import asyncio
import functools
import hashlib
import io
import json
import os
import pathlib
import sys
import uuid
from typing import (
import bokeh
import js
import param
import pyodide # isort: split
from bokeh import __version__
from bokeh.core.serialization import Buffer, Serialized, Serializer
from bokeh.document import Document
from bokeh.document.json import PatchJson
from bokeh.embed.elements import script_for_render_items
from bokeh.embed.util import standalone_docs_json_and_render_items
from bokeh.embed.wrappers import wrap_in_script_tag
from bokeh.events import DocumentReady
from bokeh.io.doc import set_curdoc
from bokeh.model import Model
from bokeh.settings import settings as bk_settings
from bokeh.util.sampledata import (
from js import JSON, XMLHttpRequest
from ..config import config
from ..util import edit_readonly, isurl
from . import resources
from .document import MockSessionContext
from .loading import LOADING_INDICATOR_CSS_CLASS
from .mime_render import WriteCallbackStream, exec_with_return, format_mime
from .state import state
def _link_docs(pydoc: Document, jsdoc: Any) -> None:
    """
    Links Python and JS documents in Pyodide ensuring that messages
    are passed between them.

    Arguments
    ---------
    pydoc: bokeh.document.Document
        The Python Bokeh Document instance to sync.
    jsdoc: Javascript Document
        The Javascript Bokeh Document instance to sync.
    """

    def jssync(event):
        setter_id = getattr(event, 'setter_id', None)
        if setter_id is not None and setter_id == 'python' or _patching:
            return
        json_patch = jsdoc.create_json_patch(pyodide.ffi.to_js([event]))
        patch = _convert_json_patch(json_patch)
        pydoc.apply_json_patch(patch, setter='js')
    jsdoc.on_change(pyodide.ffi.create_proxy(jssync), pyodide.ffi.to_js(False))

    def pysync(event):
        global _patching
        setter = getattr(event, 'setter', None)
        if setter is not None and setter == 'js':
            return
        json_patch, buffer_map = _process_document_events(pydoc, [event])
        json_patch = pyodide.ffi.to_js(json_patch, dict_converter=_dict_converter)
        buffer_map = pyodide.ffi.to_js(buffer_map)
        _patching = True
        try:
            jsdoc.apply_json_patch(json_patch, buffer_map)
        finally:
            _patching = False
    pydoc.on_change(pysync)
    try:
        pydoc.unhold()
        pydoc.callbacks.trigger_event(DocumentReady())
    except Exception as e:
        print(f'Error raised while processing Document events: {e}')