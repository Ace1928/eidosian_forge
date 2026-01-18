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
def _sock_recv(self, fut, sock, n):
    if fut.done():
        return
    try:
        data = sock.recv(n)
    except (BlockingIOError, InterruptedError):
        return
    except (SystemExit, KeyboardInterrupt):
        raise
    except BaseException as exc:
        fut.set_exception(exc)
    else:
        fut.set_result(data)