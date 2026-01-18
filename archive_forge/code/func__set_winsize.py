from __future__ import annotations
import os
import select
import shlex
import signal
import subprocess
import sys
from typing import ClassVar, Mapping
import param
from pyviz_comms import JupyterComm
from ..io.callbacks import PeriodicCallback
from ..util import edit_readonly, lazy_load
from .base import Widget
@param.depends('_terminal.ncols', '_terminal.nrows', watch=True)
def _set_winsize(self):
    if self._fd is None or not self._terminal.nrows or (not self._terminal.ncols):
        return
    import fcntl
    import struct
    import termios
    winsize = struct.pack('HHHH', self._terminal.nrows, self._terminal.ncols, 0, 0)
    try:
        fcntl.ioctl(self._fd, termios.TIOCSWINSZ, winsize)
    except OSError:
        pass