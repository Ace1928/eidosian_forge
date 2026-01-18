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
@default('shell_class')
def _default_shell_class(self):
    return InProcessInteractiveShell