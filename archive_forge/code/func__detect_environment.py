from __future__ import annotations
import numbers
import sys
from contextlib import contextmanager
from functools import wraps
from importlib import metadata as importlib_metadata
from io import UnsupportedOperation
from kombu.exceptions import reraise
def _detect_environment():
    if 'eventlet' in sys.modules:
        try:
            import socket
            from eventlet.patcher import is_monkey_patched as is_eventlet
            if is_eventlet(socket):
                return 'eventlet'
        except ImportError:
            pass
    if 'gevent' in sys.modules:
        try:
            import socket
            from gevent import socket as _gsocket
            if socket.socket is _gsocket.socket:
                return 'gevent'
        except ImportError:
            pass
    return 'default'