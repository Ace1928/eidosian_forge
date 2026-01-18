import sys
import warnings
from eventlet import greenpool
from eventlet import greenthread
from eventlet import support
from eventlet.green import socket
from eventlet.support import greenlets as greenlet
class StopServe(Exception):
    """Exception class used for quitting :func:`~eventlet.serve` gracefully."""
    pass