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
@observe('shell_streams')
def _shell_streams_changed(self, change):
    warnings.warn('Kernel.shell_streams is deprecated in ipykernel 6.0. Use Kernel.shell_stream', DeprecationWarning, stacklevel=2)
    if len(change.new) > 1:
        warnings.warn('Kernel only supports one shell stream. Additional streams will be ignored.', RuntimeWarning, stacklevel=2)
    if change.new:
        self.shell_stream = change.new[0]