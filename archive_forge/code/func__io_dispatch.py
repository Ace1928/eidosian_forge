import logging
import sys
from contextlib import contextmanager
from IPython.core.interactiveshell import InteractiveShellABC
from traitlets import Any, Enum, Instance, List, Type, default
from ipykernel.ipkernel import IPythonKernel
from ipykernel.jsonutil import json_clean
from ipykernel.zmqshell import ZMQInteractiveShell
from ..iostream import BackgroundSocket, IOPubThread, OutStream
from .constants import INPROCESS_KEY
from .socket import DummySocket
def _io_dispatch(self, change):
    """Called when a message is sent to the IO socket."""
    assert self.iopub_socket.io_thread is not None
    assert self.session is not None
    ident, msg = self.session.recv(self.iopub_socket.io_thread.socket, copy=False)
    for frontend in self.frontends:
        assert frontend is not None
        frontend.iopub_channel.call_handlers(msg)