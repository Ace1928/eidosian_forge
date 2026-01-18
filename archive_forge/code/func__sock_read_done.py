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
def _sock_read_done(self, fd, fut, handle=None):
    if handle is None or not handle.cancelled():
        self.remove_reader(fd)