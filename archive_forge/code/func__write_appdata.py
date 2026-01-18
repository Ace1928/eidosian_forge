import collections
import enum
import warnings
from . import constants
from . import exceptions
from . import protocols
from . import transports
from .log import logger
def _write_appdata(self, list_of_data):
    if self._state in (SSLProtocolState.FLUSHING, SSLProtocolState.SHUTDOWN, SSLProtocolState.UNWRAPPED):
        if self._conn_lost >= constants.LOG_THRESHOLD_FOR_CONNLOST_WRITES:
            logger.warning('SSL connection is closed')
        self._conn_lost += 1
        return
    for data in list_of_data:
        self._write_backlog.append(data)
        self._write_buffer_size += len(data)
    try:
        if self._state == SSLProtocolState.WRAPPED:
            self._do_write()
    except Exception as ex:
        self._fatal_error(ex, 'Fatal error on SSL protocol')