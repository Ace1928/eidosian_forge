from __future__ import annotations
import json
import os
import sys
import uuid
import warnings
from contextlib import contextmanager
from functools import partial
from typing import (
import bokeh
import bokeh.embed.notebook
import param
from bokeh.core.json_encoder import serialize_json
from bokeh.core.templates import MACROS
from bokeh.document import Document
from bokeh.embed import server_document
from bokeh.embed.elements import div_for_render_item, script_for_render_items
from bokeh.embed.util import standalone_docs_json_and_render_items
from bokeh.embed.wrappers import wrap_in_script_tag
from bokeh.models import Model
from bokeh.resources import CDN, INLINE
from bokeh.settings import _Unset, settings
from bokeh.util.serialization import make_id
from param.display import (
from pyviz_comms import (
from ..util import escape
from .embed import embed_state
from .model import add_to_doc, diff
from .resources import (
from .state import state
class JupyterCommJSBinary(JupyterCommJS):
    """
    Extends pyviz_comms.JupyterCommJS with support for repacking
    binary buffers.
    """

    @classmethod
    def decode(cls, msg):
        buffers = {i: v for i, v in enumerate(msg['buffers'])}
        return dict(msg['content']['data'], _buffers=buffers)