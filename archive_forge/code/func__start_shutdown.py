import collections
import enum
import warnings
from . import constants
from . import exceptions
from . import protocols
from . import transports
from .log import logger
def _start_shutdown(self):
    if self._state in (SSLProtocolState.FLUSHING, SSLProtocolState.SHUTDOWN, SSLProtocolState.UNWRAPPED):
        return
    if self._app_transport is not None:
        self._app_transport._closed = True
    if self._state == SSLProtocolState.DO_HANDSHAKE:
        self._abort(None)
    else:
        self._set_state(SSLProtocolState.FLUSHING)
        self._shutdown_timeout_handle = self._loop.call_later(self._ssl_shutdown_timeout, lambda: self._check_shutdown_timeout())
        self._do_flush()