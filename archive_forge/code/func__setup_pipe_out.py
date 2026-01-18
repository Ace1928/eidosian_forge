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
def _setup_pipe_out(self):
    ctx = zmq.Context()
    pipe_out = ctx.socket(zmq.PUSH)
    pipe_out.linger = 3000
    pipe_out.connect('tcp://127.0.0.1:%i' % self._pipe_port)
    return (ctx, pipe_out)