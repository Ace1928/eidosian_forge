import asyncio
import atexit
import contextvars
import io
import os
import sys
import threading
import traceback
import warnings
from binascii import b2a_hex
from collections import defaultdict, deque
from io import StringIO, TextIOBase
from threading import local
from typing import Any, Callable, Deque, Dict, Optional
import zmq
from jupyter_client.session import extract_header
from tornado.ioloop import IOLoop
from zmq.eventloop.zmqstream import ZMQStream
class BackgroundSocket:
    """Wrapper around IOPub thread that provides zmq send[_multipart]"""
    io_thread = None

    def __init__(self, io_thread):
        """Initialize the socket."""
        self.io_thread = io_thread

    def __getattr__(self, attr):
        """Wrap socket attr access for backward-compatibility"""
        if attr.startswith('__') and attr.endswith('__'):
            super().__getattr__(attr)
        assert self.io_thread is not None
        if hasattr(self.io_thread.socket, attr):
            warnings.warn(f'Accessing zmq Socket attribute {attr} on BackgroundSocket is deprecated since ipykernel 4.3.0 use .io_thread.socket.{attr}', DeprecationWarning, stacklevel=2)
            return getattr(self.io_thread.socket, attr)
        return super().__getattr__(attr)

    def __setattr__(self, attr, value):
        """Set an attribute on the socket."""
        if attr == 'io_thread' or (attr.startswith('__') and attr.endswith('__')):
            super().__setattr__(attr, value)
        else:
            warnings.warn(f'Setting zmq Socket attribute {attr} on BackgroundSocket is deprecated since ipykernel 4.3.0 use .io_thread.socket.{attr}', DeprecationWarning, stacklevel=2)
            assert self.io_thread is not None
            setattr(self.io_thread.socket, attr, value)

    def send(self, msg, *args, **kwargs):
        """Send a message to the socket."""
        return self.send_multipart([msg], *args, **kwargs)

    def send_multipart(self, *args, **kwargs):
        """Schedule send in IO thread"""
        assert self.io_thread is not None
        return self.io_thread.send_multipart(*args, **kwargs)