import asyncio
import builtins
import gc
import getpass
import os
import signal
import sys
import threading
import typing as t
from contextlib import contextmanager
from functools import partial
import comm
from IPython.core import release
from IPython.utils.tokenutil import line_at_cursor, token_at_cursor
from jupyter_client.session import extract_header
from traitlets import Any, Bool, HasTraits, Instance, List, Type, observe, observe_compat
from zmq.eventloop.zmqstream import ZMQStream
from .comm.comm import BaseComm
from .comm.manager import CommManager
from .compiler import XCachingCompiler
from .eventloops import _use_appnope
from .iostream import OutStream
from .kernelbase import Kernel as KernelBase
from .kernelbase import _accepts_parameters
from .zmqshell import ZMQInteractiveShell
def _forward_input(self, allow_stdin=False):
    """Forward raw_input and getpass to the current frontend.

        via input_request
        """
    self._allow_stdin = allow_stdin
    self._sys_raw_input = builtins.input
    builtins.input = self.raw_input
    self._save_getpass = getpass.getpass
    getpass.getpass = self.getpass