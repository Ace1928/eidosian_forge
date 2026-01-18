import collections
import datetime
import heapq
from tornado import gen, ioloop
from tornado.concurrent import Future, future_set_result_unless_cancelled
from tornado.locks import Event
from typing import Union, TypeVar, Generic, Awaitable, Optional
import typing
class _QueueIterator(Generic[_T]):

    def __init__(self, q: 'Queue[_T]') -> None:
        self.q = q

    def __anext__(self) -> Awaitable[_T]:
        return self.q.get()