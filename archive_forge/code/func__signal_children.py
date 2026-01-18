from __future__ import annotations
import asyncio
import concurrent.futures
import inspect
import itertools
import logging
import os
import socket
import sys
import threading
import time
import typing as t
import uuid
import warnings
from datetime import datetime
from functools import partial
from signal import SIGINT, SIGTERM, Signals, default_int_handler, signal
from .control import CONTROL_THREAD_NAME
import psutil
import zmq
from IPython.core.error import StdinNotImplementedError
from jupyter_client.session import Session
from tornado import ioloop
from tornado.queues import Queue, QueueEmpty
from traitlets.config.configurable import SingletonConfigurable
from traitlets.traitlets import (
from zmq.eventloop.zmqstream import ZMQStream
from ipykernel.jsonutil import json_clean
from ._version import kernel_protocol_version
from .iostream import OutStream
def _signal_children(self, signum):
    """
        Send a signal to all our children

        Like `killpg`, but does not include the current process
        (or possible parents).
        """
    sig_rep = f'{Signals(signum)!r}'
    for p in self._process_children():
        self.log.debug('Sending %s to subprocess %s', sig_rep, p)
        try:
            if signum == SIGTERM:
                p.terminate()
            elif signum == SIGKILL:
                p.kill()
            else:
                p.send_signal(signum)
        except psutil.NoSuchProcess:
            pass