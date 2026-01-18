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
def _forward_subprocess_output_to_terminal(self):
    if not self._fd:
        return
    data_ready, _, _ = select.select([self._fd], [], [], self._timeout_sec)
    if not data_ready:
        return
    output = self._decode_utf8_on_boundary(self._fd, self._max_read_bytes)
    if 'CompletedProcess' in output:
        self._reset()
        output = self._remove_last_line_from_string(output)
    self._terminal.write(output)