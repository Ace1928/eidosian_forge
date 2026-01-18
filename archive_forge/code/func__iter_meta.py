import datetime
import time
from collections import deque
from contextlib import contextmanager
from weakref import proxy
from dateutil.parser import isoparse
from kombu.utils.objects import cached_property
from vine import Thenable, barrier, promise
from . import current_app, states
from ._state import _set_task_join_will_block, task_join_will_block
from .app import app_or_default
from .exceptions import ImproperlyConfigured, IncompleteStream, TimeoutError
from .utils.graph import DependencyGraph, GraphFormatter
def _iter_meta(self, **kwargs):
    return (meta for _, meta in self.backend.get_many({r.id for r in self.results}, max_iterations=1, **kwargs))