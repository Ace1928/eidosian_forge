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
def _flush_buffers(self):
    """clear the current buffer and return the current buffer data."""
    buffers = self._rotate_buffers()
    for frozen_parent, buffer in buffers.items():
        data = buffer.getvalue()
        buffer.close()
        yield (dict(frozen_parent), data)