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
def _send_abort_reply(self, stream, msg, idents):
    """Send a reply to an aborted request"""
    if not self.session:
        return
    self.log.info('Aborting %s: %s', msg['header']['msg_id'], msg['header']['msg_type'])
    reply_type = msg['header']['msg_type'].rsplit('_', 1)[0] + '_reply'
    status = {'status': 'aborted'}
    md = self.init_metadata(msg)
    md = self.finish_metadata(msg, md, status)
    md.update(status)
    self.session.send(stream, reply_type, metadata=md, content=status, parent=msg, ident=idents)