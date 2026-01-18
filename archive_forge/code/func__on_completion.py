import concurrent.futures
import contextvars
import functools
import inspect
import itertools
import types
import warnings
import weakref
from types import GenericAlias
from . import base_tasks
from . import coroutines
from . import events
from . import exceptions
from . import futures
from .coroutines import _is_coroutine
def _on_completion(f):
    if not todo:
        return
    todo.remove(f)
    done.put_nowait(f)
    if not todo and timeout_handle is not None:
        timeout_handle.cancel()