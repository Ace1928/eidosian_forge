import ast
import asyncio
import inspect
from functools import wraps
def _pseudo_sync_runner(coro):
    """
    A runner that does not really allow async execution, and just advance the coroutine.

    See discussion in https://github.com/python-trio/trio/issues/608,

    Credit to Nathaniel Smith
    """
    try:
        coro.send(None)
    except StopIteration as exc:
        return exc.value
    else:
        raise RuntimeError('{coro_name!r} needs a real async loop'.format(coro_name=coro.__name__))