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
class InProcessInteractiveShell(ZMQInteractiveShell):
    """An in-process interactive shell."""
    kernel: InProcessKernel = Instance('ipykernel.inprocess.ipkernel.InProcessKernel', allow_none=True)

    def enable_gui(self, gui=None):
        """Enable GUI integration for the kernel."""
        if not gui:
            gui = self.kernel.gui
        self.active_eventloop = gui

    def enable_matplotlib(self, gui=None):
        """Enable matplotlib integration for the kernel."""
        if not gui:
            gui = self.kernel.gui
        return super().enable_matplotlib(gui)

    def enable_pylab(self, gui=None, import_all=True, welcome_message=False):
        """Activate pylab support at runtime."""
        if not gui:
            gui = self.kernel.gui
        return super().enable_pylab(gui, import_all, welcome_message)