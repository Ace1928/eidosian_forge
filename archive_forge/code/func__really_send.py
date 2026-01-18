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
def _really_send(self, msg, *args, **kwargs):
    """The callback that actually sends messages"""
    if self.closed:
        return
    mp_mode = self._check_mp_mode()
    if mp_mode != CHILD:
        self.socket.send_multipart(msg, *args, **kwargs)
    else:
        ctx, pipe_out = self._setup_pipe_out()
        pipe_out.send_multipart([self._pipe_uuid, *msg], *args, **kwargs)
        pipe_out.close()
        ctx.term()