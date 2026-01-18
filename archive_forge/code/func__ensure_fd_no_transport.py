import collections
import errno
import functools
import selectors
import socket
import warnings
import weakref
from . import base_events
from . import constants
from . import events
from . import futures
from . import protocols
from . import sslproto
from . import transports
from . import trsock
from .log import logger
def _ensure_fd_no_transport(self, fd):
    fileno = fd
    if not isinstance(fileno, int):
        try:
            fileno = int(fileno.fileno())
        except (AttributeError, TypeError, ValueError):
            raise ValueError(f'Invalid file object: {fd!r}') from None
    try:
        transport = self._transports[fileno]
    except KeyError:
        pass
    else:
        if not transport.is_closing():
            raise RuntimeError(f'File descriptor {fd!r} is used by transport {transport!r}')