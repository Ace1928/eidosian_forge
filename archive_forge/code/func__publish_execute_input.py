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
def _publish_execute_input(self, code, parent, execution_count):
    """Publish the code request on the iopub stream."""
    if not self.session:
        return
    self.session.send(self.iopub_socket, 'execute_input', {'code': code, 'execution_count': execution_count}, parent=parent, ident=self._topic('execute_input'))