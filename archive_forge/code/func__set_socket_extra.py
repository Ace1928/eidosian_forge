import io
import os
import socket
import warnings
import signal
import threading
import collections
from . import base_events
from . import constants
from . import futures
from . import exceptions
from . import protocols
from . import sslproto
from . import transports
from . import trsock
from .log import logger
def _set_socket_extra(transport, sock):
    transport._extra['socket'] = trsock.TransportSocket(sock)
    try:
        transport._extra['sockname'] = sock.getsockname()
    except socket.error:
        if transport._loop.get_debug():
            logger.warning('getsockname() failed on %r', sock, exc_info=True)
    if 'peername' not in transport._extra:
        try:
            transport._extra['peername'] = sock.getpeername()
        except socket.error:
            transport._extra['peername'] = None