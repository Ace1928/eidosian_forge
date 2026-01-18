from __future__ import annotations
import re
import sys
from contextlib import contextmanager
from functools import partial
from io import BytesIO
from typing import (
import param
from bokeh.models import (
from bokeh.themes import Theme
from ..io import remove_root, state
from ..io.notebook import push
from ..util import escape
from ..viewable import Layoutable
from .base import PaneBase
from .image import (
from .ipywidget import IPyWidget
from .markup import HTML
@classmethod
def _property_callback_wrapper(cls, cb, doc, comm, callbacks):

    def wrapped_callback(attr, old, new):
        with _wrap_callback(cb, wrapped_callback, doc, comm, callbacks):
            cb(attr, old, new)
    return wrapped_callback