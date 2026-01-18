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
def _setup_stream_redirects(self, name):
    pr, pw = os.pipe()
    fno = self._original_stdstream_fd = getattr(sys, name).fileno()
    self._original_stdstream_copy = os.dup(fno)
    os.dup2(pw, fno)
    self._fid = pr
    self._exc = None
    self.watch_fd_thread = threading.Thread(target=self._watch_pipe_fd)
    self.watch_fd_thread.daemon = True
    self.watch_fd_thread.start()