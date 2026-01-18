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
def _decode_utf8_on_boundary(self, fd, max_read_bytes, max_extra_bytes=2):
    """UTF-8 characters can be multi-byte so need to decode on correct boundary"""
    data = os.read(fd, max_read_bytes)
    for _ in range(max_extra_bytes + 1):
        try:
            return data.decode('utf-8')
        except UnicodeDecodeError:
            data = data + os.read(fd, 1)
    raise UnicodeError('Could not find decode boundary for UTF-8')