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
def _wrap_bokeh_callbacks(cls, root, bokeh_model, doc, comm):
    for model in bokeh_model.select({'type': Model}):
        for key, cbs in model._callbacks.items():
            callbacks = model._callbacks[key]
            callbacks[:] = [cls._property_callback_wrapper(cb, doc, comm, callbacks) for cb in cbs]
        for key, cbs in model._event_callbacks.items():
            callbacks = model._event_callbacks[key]
            callbacks[:] = [cls._event_callback_wrapper(cb, doc, comm, callbacks) for cb in cbs]