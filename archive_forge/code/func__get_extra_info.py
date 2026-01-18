import collections
import enum
import warnings
from . import constants
from . import exceptions
from . import protocols
from . import transports
from .log import logger
def _get_extra_info(self, name, default=None):
    if name in self._extra:
        return self._extra[name]
    elif self._transport is not None:
        return self._transport.get_extra_info(name, default)
    else:
        return default