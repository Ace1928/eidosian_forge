from __future__ import annotations
import logging # isort:skip
import sys
import threading
from collections import defaultdict
from traceback import format_exception
from typing import (
import tornado
from tornado import gen
from ..core.types import ID
class _AsyncPeriodic:
    """ Like ioloop.PeriodicCallback except the 'func' can be async and return
    a Future.

    Will wait for func to finish each time before we call it again. (Plain
    ioloop.PeriodicCallback can "pile up" invocations if they are taking too
    long.)

    """
    _loop: IOLoop
    _period: int
    _started: bool
    _stopped: bool

    def __init__(self, func: Callback, period: int, io_loop: IOLoop) -> None:
        self._func: Callback = func
        self._loop = io_loop
        self._period = period
        self._started = False
        self._stopped = False

    def sleep(self) -> gen.Future[None]:
        f: gen.Future[None] = gen.Future()
        self._loop.call_later(self._period / 1000.0, lambda: f.set_result(None))
        return f

    def start(self) -> None:
        if self._started:
            raise RuntimeError('called start() twice on _AsyncPeriodic')
        self._started = True

        def invoke() -> InvokeResult:
            sleep_future = self.sleep()
            result = self._func()
            if result is None:
                return sleep_future
            callback_future = gen.convert_yielded(result)
            return gen.multi([sleep_future, callback_future])

        def on_done(future: gen.Future[None]) -> None:
            if not self._stopped:
                self._loop.add_future(invoke(), on_done)
            ex = future.exception()
            if ex is not None:
                log.error('Error thrown from periodic callback:')
                lines = format_exception(ex.__class__, ex, ex.__traceback__)
                log.error(''.join(lines))
        self._loop.add_future(self.sleep(), on_done)

    def stop(self) -> None:
        self._stopped = True