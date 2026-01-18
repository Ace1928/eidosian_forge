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
def _link_docs_worker(doc: Document, dispatch_fn: Any, msg_id: str | None=None, setter: str | None=None):
    """
    Links the Python document to a dispatch_fn which can be used to
    sync messages between a WebWorker and the main thread in the
    browser.

    Arguments
    ---------
    doc: bokeh.document.Document
        The document to dispatch messages from.
    dispatch_fn: JS function
        The Javascript function to dispatch messages to.
    setter: str
        Setter ID used for suppressing events.
    msg_id: str | None
        An optional message ID to pass through to the dispatch_fn.
    """

    def pysync(event):
        if setter is not None and getattr(event, 'setter', None) == setter:
            return
        json_patch, buffer_map = _process_document_events(doc, [event])
        json_patch = pyodide.ffi.to_js(json_patch, dict_converter=_dict_converter)
        dispatch_fn(json_patch, pyodide.ffi.to_js(buffer_map), msg_id)
    doc.on_change(pysync)
    doc.unhold()
    doc.callbacks.trigger_event(DocumentReady())