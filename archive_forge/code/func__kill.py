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
def _kill(self, *events):
    child_pid = self._child_pid
    self._reset()
    if child_pid:
        os.killpg(os.getpgid(child_pid), signal.SIGTERM)
        self._terminal.write(f'\nThe process {child_pid} was killed\n')
    else:
        self._terminal.write('\nNo running process to kill\n')