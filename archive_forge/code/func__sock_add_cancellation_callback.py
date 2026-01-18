import errno
import io
import itertools
import os
import selectors
import signal
import socket
import stat
import subprocess
import sys
import threading
import warnings
from . import base_events
from . import base_subprocess
from . import constants
from . import coroutines
from . import events
from . import exceptions
from . import futures
from . import selector_events
from . import tasks
from . import transports
from .log import logger
def _sock_add_cancellation_callback(self, fut, sock):

    def cb(fut):
        if fut.cancelled():
            fd = sock.fileno()
            if fd != -1:
                self.remove_writer(fd)
    fut.add_done_callback(cb)