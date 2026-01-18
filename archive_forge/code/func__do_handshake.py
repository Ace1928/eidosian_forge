import collections
import enum
import warnings
from . import constants
from . import exceptions
from . import protocols
from . import transports
from .log import logger
def _do_handshake(self):
    try:
        self._sslobj.do_handshake()
    except SSLAgainErrors:
        self._process_outgoing()
    except ssl.SSLError as exc:
        self._on_handshake_complete(exc)
    else:
        self._on_handshake_complete(None)