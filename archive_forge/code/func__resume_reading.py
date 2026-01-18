import collections
import enum
import warnings
from . import constants
from . import exceptions
from . import protocols
from . import transports
from .log import logger
def _resume_reading(self):
    if self._app_reading_paused:
        self._app_reading_paused = False

        def resume():
            if self._state == SSLProtocolState.WRAPPED:
                self._do_read()
            elif self._state == SSLProtocolState.FLUSHING:
                self._do_flush()
            elif self._state == SSLProtocolState.SHUTDOWN:
                self._do_shutdown()
        self._loop.call_soon(resume)