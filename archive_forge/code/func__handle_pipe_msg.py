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
def _handle_pipe_msg(self, msg):
    """handle a pipe message from a subprocess"""
    if not self._pipe_flag or not self._is_master_process():
        return
    if msg[0] != self._pipe_uuid:
        print('Bad pipe message: %s', msg, file=sys.__stderr__)
        return
    self.send_multipart(msg[1:])