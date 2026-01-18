import collections
import enum
import warnings
from . import constants
from . import exceptions
from . import protocols
from . import transports
from .log import logger
def _on_shutdown_complete(self, shutdown_exc):
    if self._shutdown_timeout_handle is not None:
        self._shutdown_timeout_handle.cancel()
        self._shutdown_timeout_handle = None
    if shutdown_exc:
        self._fatal_error(shutdown_exc)
    else:
        self._loop.call_soon(self._transport.close)