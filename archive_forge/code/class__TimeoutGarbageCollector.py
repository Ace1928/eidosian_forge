import collections
import datetime
import types
from tornado import gen, ioloop
from tornado.concurrent import Future, future_set_result_unless_cancelled
from typing import Union, Optional, Type, Any, Awaitable
import typing
class _TimeoutGarbageCollector(object):
    """Base class for objects that periodically clean up timed-out waiters.

    Avoids memory leak in a common pattern like:

        while True:
            yield condition.wait(short_timeout)
            print('looping....')
    """

    def __init__(self) -> None:
        self._waiters = collections.deque()
        self._timeouts = 0

    def _garbage_collect(self) -> None:
        self._timeouts += 1
        if self._timeouts > 100:
            self._timeouts = 0
            self._waiters = collections.deque((w for w in self._waiters if not w.done()))