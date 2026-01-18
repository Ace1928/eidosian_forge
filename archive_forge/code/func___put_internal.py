import collections
import datetime
import heapq
from tornado import gen, ioloop
from tornado.concurrent import Future, future_set_result_unless_cancelled
from tornado.locks import Event
from typing import Union, TypeVar, Generic, Awaitable, Optional
import typing
def __put_internal(self, item: _T) -> None:
    self._unfinished_tasks += 1
    self._finished.clear()
    self._put(item)