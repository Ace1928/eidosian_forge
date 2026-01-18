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
def _schedule_flush(self):
    """schedule a flush in the IO thread

        call this on write, to indicate that flush should be called soon.
        """
    if self._flush_pending:
        return
    self._flush_pending = True

    def _schedule_in_thread():
        self._io_loop.call_later(self.flush_interval, self._flush)
    self.pub_thread.schedule(_schedule_in_thread)