from __future__ import annotations
import asyncio
import dataclasses
import datetime as dt
import gc
import inspect
import json
import logging
import sys
import threading
import time
import weakref
from contextlib import contextmanager
from functools import partial, wraps
from typing import (
from bokeh.application.application import SessionContext
from bokeh.core.serialization import Serializable
from bokeh.document.document import Document
from bokeh.document.events import (
from bokeh.model.util import visit_immediate_value_references
from bokeh.models import CustomJS
from ..config import config
from ..util import param_watchers
from .loading import LOADING_INDICATOR_CSS_CLASS
from .model import hold, monkeypatch_events  # noqa: F401 API import
from .state import curdoc_locked, state
def _destroy_document(self, session):
    """
    Override for Document.destroy() without calling gc.collect directly.
    The gc.collect() call is scheduled as a task, ensuring that when
    multiple documents are destroyed in quick succession we do not
    schedule excessive garbage collection.
    """
    if session is not None:
        self.remove_on_change(session)
    del self._roots
    del self._theme
    del self._template
    self._session_context = None
    self.callbacks.destroy()
    self.models.destroy()
    for module in self.modules._modules:
        if module.__name__ in sys.modules:
            del sys.modules[module.__name__]
        module.__dict__.clear()
        del module
    self.modules._modules = []
    for cb in state._periodic.get(self, []):
        cb.stop()
    for attr in dir(state):
        if not attr.startswith('_') or attr == '_param_watchers':
            continue
        state_obj = getattr(state, attr)
        if isinstance(state_obj, weakref.WeakKeyDictionary) and self in state_obj:
            del state_obj[self]
    global _panel_last_cleanup
    _panel_last_cleanup = time.monotonic()
    at = dt.datetime.now() + dt.timedelta(seconds=GC_DEBOUNCE)
    state.schedule_task('gc.collect', _garbage_collect, at=at)
    del self.destroy