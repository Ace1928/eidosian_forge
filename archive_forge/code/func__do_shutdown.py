import collections
import enum
import warnings
from . import constants
from . import exceptions
from . import protocols
from . import transports
from .log import logger
def _do_shutdown(self):
    try:
        if not self._eof_received:
            self._sslobj.unwrap()
    except SSLAgainErrors:
        self._process_outgoing()
    except ssl.SSLError as exc:
        self._on_shutdown_complete(exc)
    else:
        self._process_outgoing()
        self._call_eof_received()
        self._on_shutdown_complete(None)