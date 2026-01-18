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
def _cleanup_task(task):
    if task in _write_tasks:
        _write_tasks.remove(task)