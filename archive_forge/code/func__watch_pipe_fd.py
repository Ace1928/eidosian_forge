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
def _watch_pipe_fd(self):
    """
        We've redirected standards streams 0 and 1 into a pipe.

        We need to watch in a thread and redirect them to the right places.

        1) the ZMQ channels to show in notebook interfaces,
        2) the original stdout/err, to capture errors in terminals.

        We cannot schedule this on the ioloop thread, as this might be blocking.

        """
    try:
        bts = os.read(self._fid, PIPE_BUFFER_SIZE)
        while bts and self._should_watch:
            self.write(bts.decode(errors='replace'))
            os.write(self._original_stdstream_copy, bts)
            bts = os.read(self._fid, PIPE_BUFFER_SIZE)
    except Exception:
        self._exc = sys.exc_info()