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
def _bytes_converter(value, converter, other):
    if not hasattr(value, 'buffer'):
        return value
    value = dict(value.object_entries())
    uid = uuid.uuid4().hex
    _current_buffers.append(Buffer(id=uid, data=value['buffer'].to_bytes()))
    return {'id': uid}