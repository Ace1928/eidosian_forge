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
def _test_selector_event(selector, fd, event):
    try:
        key = selector.get_key(fd)
    except KeyError:
        return False
    else:
        return bool(key.events & event)