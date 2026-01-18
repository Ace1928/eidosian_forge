import asyncio
import builtins
import collections
from collections.abc import Generator
import concurrent.futures
import datetime
import functools
from functools import singledispatch
from inspect import isawaitable
import sys
import types
from tornado.concurrent import (
from tornado.ioloop import IOLoop
from tornado.log import app_log
from tornado.util import TimeoutError
import typing
from typing import Union, Any, Callable, List, Type, Tuple, Awaitable, Dict, overload
def _return_result(self, done: Future) -> Future:
    """Called set the returned future's state that of the future
        we yielded, and set the current future for the iterator.
        """
    if self._running_future is None:
        raise Exception('no future is running')
    chain_future(done, self._running_future)
    res = self._running_future
    self._running_future = None
    self.current_future = done
    self.current_index = self._unfinished.pop(done)
    return res