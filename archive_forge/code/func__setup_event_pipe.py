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
def _setup_event_pipe(self):
    """Create the PULL socket listening for events that should fire in this thread."""
    ctx = self.socket.context
    pipe_in = ctx.socket(zmq.PULL)
    pipe_in.linger = 0
    _uuid = b2a_hex(os.urandom(16)).decode('ascii')
    iface = self._event_interface = 'inproc://%s' % _uuid
    pipe_in.bind(iface)
    self._event_puller = ZMQStream(pipe_in, self.io_loop)
    self._event_puller.on_recv(self._handle_event)