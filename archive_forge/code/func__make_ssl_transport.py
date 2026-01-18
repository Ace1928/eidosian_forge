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
def _make_ssl_transport(self, rawsock, protocol, sslcontext, waiter=None, *, server_side=False, server_hostname=None, extra=None, server=None, ssl_handshake_timeout=None, ssl_shutdown_timeout=None):
    ssl_protocol = sslproto.SSLProtocol(self, protocol, sslcontext, waiter, server_side, server_hostname, ssl_handshake_timeout=ssl_handshake_timeout, ssl_shutdown_timeout=ssl_shutdown_timeout)
    _ProactorSocketTransport(self, rawsock, ssl_protocol, extra=extra, server=server)
    return ssl_protocol._app_transport