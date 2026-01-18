import socket
from . import coroutines
from . import events
from . import futures
from . import protocols
from .coroutines import coroutine
from .log import logger
@coroutine
def _drain_helper(self):
    if self._connection_lost:
        raise ConnectionResetError('Connection lost')
    if not self._paused:
        return
    waiter = self._drain_waiter
    assert waiter is None or waiter.cancelled()
    waiter = futures.Future(loop=self._loop)
    self._drain_waiter = waiter
    yield from waiter