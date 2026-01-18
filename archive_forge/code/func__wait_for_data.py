import socket
from . import coroutines
from . import events
from . import futures
from . import protocols
from .coroutines import coroutine
from .log import logger
def _wait_for_data(self, func_name):
    """Wait until feed_data() or feed_eof() is called."""
    if self._waiter is not None:
        raise RuntimeError('%s() called while another coroutine is already waiting for incoming data' % func_name)
    self._waiter = futures.Future(loop=self._loop)
    try:
        yield from self._waiter
    finally:
        self._waiter = None