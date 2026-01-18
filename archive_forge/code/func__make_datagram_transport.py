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
def _make_datagram_transport(self, sock, protocol, address=None, waiter=None, extra=None):
    return _ProactorDatagramTransport(self, sock, protocol, address, waiter, extra)