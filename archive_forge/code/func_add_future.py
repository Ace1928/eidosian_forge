import asyncio
import concurrent.futures
import datetime
import functools
import numbers
import os
import sys
import time
import math
import random
import warnings
from inspect import isawaitable
from tornado.concurrent import (
from tornado.log import app_log
from tornado.util import Configurable, TimeoutError, import_object
import typing
from typing import Union, Any, Type, Optional, Callable, TypeVar, Tuple, Awaitable
def add_future(self, future: 'Union[Future[_T], concurrent.futures.Future[_T]]', callback: Callable[['Future[_T]'], None]) -> None:
    """Schedules a callback on the ``IOLoop`` when the given
        `.Future` is finished.

        The callback is invoked with one argument, the
        `.Future`.

        This method only accepts `.Future` objects and not other
        awaitables (unlike most of Tornado where the two are
        interchangeable).
        """
    if isinstance(future, Future):
        future.add_done_callback(lambda f: self._run_callback(functools.partial(callback, f)))
    else:
        assert is_future(future)
        future_add_done_callback(future, lambda f: self.add_callback(callback, f))