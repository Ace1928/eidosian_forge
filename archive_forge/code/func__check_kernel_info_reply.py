import asyncio
import atexit
import time
from concurrent.futures import Future
from functools import partial
from threading import Thread
from typing import Any, Dict, List, Optional
import zmq
from tornado.ioloop import IOLoop
from traitlets import Instance, Type
from traitlets.log import get_logger
from zmq.eventloop import zmqstream
from .channels import HBChannel
from .client import KernelClient
from .session import Session
def _check_kernel_info_reply(self, msg: Dict[str, Any]) -> None:
    """This is run in the ioloop thread when the kernel info reply is received"""
    if msg['msg_type'] == 'kernel_info_reply':
        self._handle_kernel_info_reply(msg)
        self.shell_channel._inspect = None