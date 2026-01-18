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
@types.coroutine
def _wrap_awaitable(awaitable):
    """Helper for asyncio.ensure_future().

    Wraps awaitable (an object with __await__) into a coroutine
    that will later be wrapped in a Task by ensure_future().
    """
    return (yield from awaitable.__await__())