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
def _rotate_buffers(self):
    """Returns the current buffer and replaces it with an empty buffer."""
    with self._buffer_lock:
        old_buffers = self._buffers
        self._buffers = defaultdict(StringIO)
    return old_buffers