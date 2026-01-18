import collections
import enum
import warnings
from . import constants
from . import exceptions
from . import protocols
from . import transports
from .log import logger
def _get_write_buffer_size(self):
    return self._outgoing.pending + self._write_buffer_size