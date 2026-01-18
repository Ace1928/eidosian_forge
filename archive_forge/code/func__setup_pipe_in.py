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
def _setup_pipe_in(self):
    """setup listening pipe for IOPub from forked subprocesses"""
    ctx = self.socket.context
    self._pipe_uuid = os.urandom(16)
    pipe_in = ctx.socket(zmq.PULL)
    pipe_in.linger = 0
    try:
        self._pipe_port = pipe_in.bind_to_random_port('tcp://127.0.0.1')
    except zmq.ZMQError as e:
        warnings.warn("Couldn't bind IOPub Pipe to 127.0.0.1: %s" % e + '\nsubprocess output will be unavailable.', stacklevel=2)
        self._pipe_flag = False
        pipe_in.close()
        return
    self._pipe_in = ZMQStream(pipe_in, self.io_loop)
    self._pipe_in.on_recv(self._handle_pipe_msg)