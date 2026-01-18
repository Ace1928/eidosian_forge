import collections
import enum
import warnings
from . import constants
from . import exceptions
from . import protocols
from . import transports
from .log import logger
class SSLProtocolState(enum.Enum):
    UNWRAPPED = 'UNWRAPPED'
    DO_HANDSHAKE = 'DO_HANDSHAKE'
    WRAPPED = 'WRAPPED'
    FLUSHING = 'FLUSHING'
    SHUTDOWN = 'SHUTDOWN'