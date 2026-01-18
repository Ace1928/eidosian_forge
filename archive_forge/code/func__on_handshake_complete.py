import collections
import enum
import warnings
from . import constants
from . import exceptions
from . import protocols
from . import transports
from .log import logger
def _on_handshake_complete(self, handshake_exc):
    if self._handshake_timeout_handle is not None:
        self._handshake_timeout_handle.cancel()
        self._handshake_timeout_handle = None
    sslobj = self._sslobj
    try:
        if handshake_exc is None:
            self._set_state(SSLProtocolState.WRAPPED)
        else:
            raise handshake_exc
        peercert = sslobj.getpeercert()
    except Exception as exc:
        handshake_exc = None
        self._set_state(SSLProtocolState.UNWRAPPED)
        if isinstance(exc, ssl.CertificateError):
            msg = 'SSL handshake failed on verifying the certificate'
        else:
            msg = 'SSL handshake failed'
        self._fatal_error(exc, msg)
        self._wakeup_waiter(exc)
        return
    if self._loop.get_debug():
        dt = self._loop.time() - self._handshake_start_time
        logger.debug('%r: SSL handshake took %.1f ms', self, dt * 1000.0)
    self._extra.update(peercert=peercert, cipher=sslobj.cipher(), compression=sslobj.compression(), ssl_object=sslobj)
    if self._app_state == AppProtocolState.STATE_INIT:
        self._app_state = AppProtocolState.STATE_CON_MADE
        self._app_protocol.connection_made(self._get_app_transport())
    self._wakeup_waiter()
    self._do_read()