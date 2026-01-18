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
def _done_callback(fut):
    nonlocal nfinished
    nfinished += 1
    if outer is None or outer.done():
        if not fut.cancelled():
            fut.exception()
        return
    if not return_exceptions:
        if fut.cancelled():
            exc = fut._make_cancelled_error()
            outer.set_exception(exc)
            return
        else:
            exc = fut.exception()
            if exc is not None:
                outer.set_exception(exc)
                return
    if nfinished == nfuts:
        results = []
        for fut in children:
            if fut.cancelled():
                res = exceptions.CancelledError('' if fut._cancel_message is None else fut._cancel_message)
            else:
                res = fut.exception()
                if res is None:
                    res = fut.result()
            results.append(res)
        if outer._cancel_requested:
            exc = fut._make_cancelled_error()
            outer.set_exception(exc)
        else:
            outer.set_result(results)