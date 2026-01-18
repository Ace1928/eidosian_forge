import collections
import enum
import warnings
from . import constants
from . import exceptions
from . import protocols
from . import transports
from .log import logger
def _do_read(self):
    if self._state not in (SSLProtocolState.WRAPPED, SSLProtocolState.FLUSHING):
        return
    try:
        if not self._app_reading_paused:
            if self._app_protocol_is_buffer:
                self._do_read__buffered()
            else:
                self._do_read__copied()
            if self._write_backlog:
                self._do_write()
            else:
                self._process_outgoing()
        self._control_ssl_reading()
    except Exception as ex:
        self._fatal_error(ex, 'Fatal error on SSL protocol')