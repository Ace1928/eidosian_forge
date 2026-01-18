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
def _initialize_thread_hooks(self):
    """Store thread hierarchy and thread-parent_header associations."""
    stdout = self._stdout
    stderr = self._stderr
    kernel_thread_ident = threading.get_ident()
    kernel = self
    _threading_Thread_run = threading.Thread.run
    _threading_Thread__init__ = threading.Thread.__init__

    def run_closure(self: threading.Thread):
        """Wrap the `threading.Thread.start` to intercept thread identity.

            This is needed because there is no "start" hook yet, but there
            might be one in the future: https://bugs.python.org/issue14073

            This is a no-op if the `self._stdout` and `self._stderr` are not
            sub-classes of `OutStream`.
            """
        try:
            parent = self._ipykernel_parent_thread_ident
        except AttributeError:
            return
        for stream in [stdout, stderr]:
            if isinstance(stream, OutStream):
                if parent == kernel_thread_ident:
                    stream._thread_to_parent_header[self.ident] = kernel._new_threads_parent_header
                else:
                    stream._thread_to_parent[self.ident] = parent
        _threading_Thread_run(self)

    def init_closure(self: threading.Thread, *args, **kwargs):
        _threading_Thread__init__(self, *args, **kwargs)
        self._ipykernel_parent_thread_ident = threading.get_ident()
    threading.Thread.__init__ = init_closure
    threading.Thread.run = run_closure