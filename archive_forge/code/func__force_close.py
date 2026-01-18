import collections
import enum
import warnings
from . import constants
from . import exceptions
from . import protocols
from . import transports
from .log import logger
def _force_close(self, exc):
    self._closed = True
    if self._ssl_protocol is not None:
        self._ssl_protocol._abort(exc)