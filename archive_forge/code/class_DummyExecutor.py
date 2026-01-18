import asyncio
from concurrent import futures
import functools
import sys
import types
from tornado.log import app_log
import typing
from typing import Any, Callable, Optional, Tuple, Union
class DummyExecutor(futures.Executor):

    def submit(self, fn: Callable[..., _T], *args: Any, **kwargs: Any) -> 'futures.Future[_T]':
        future = futures.Future()
        try:
            future_set_result_unless_cancelled(future, fn(*args, **kwargs))
        except Exception:
            future_set_exc_info(future, sys.exc_info())
        return future
    if sys.version_info >= (3, 9):

        def shutdown(self, wait: bool=True, cancel_futures: bool=False) -> None:
            pass
    else:

        def shutdown(self, wait: bool=True) -> None:
            pass