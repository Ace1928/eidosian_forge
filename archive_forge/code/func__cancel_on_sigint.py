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
@contextmanager
def _cancel_on_sigint(self, future):
    """ContextManager for capturing SIGINT and cancelling a future

        SIGINT raises in the event loop when running async code,
        but we want it to halt a coroutine.

        Ideally, it would raise KeyboardInterrupt,
        but this turns it into a CancelledError.
        At least it gets a decent traceback to the user.
        """
    sigint_future: asyncio.Future[int] = asyncio.Future()

    def cancel_unless_done(f, _ignored):
        if f.cancelled() or f.done():
            return
        f.cancel()
    sigint_future.add_done_callback(partial(cancel_unless_done, future))
    future.add_done_callback(partial(cancel_unless_done, sigint_future))

    def handle_sigint(*args):

        def set_sigint_result():
            if sigint_future.cancelled() or sigint_future.done():
                return
            sigint_future.set_result(1)
        self.io_loop.add_callback(set_sigint_result)
    save_sigint = signal.signal(signal.SIGINT, handle_sigint)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, save_sigint)