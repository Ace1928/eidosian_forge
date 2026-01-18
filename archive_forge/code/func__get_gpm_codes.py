from __future__ import annotations
import contextlib
import fcntl
import functools
import os
import selectors
import signal
import struct
import sys
import termios
import tty
import typing
from subprocess import PIPE, Popen
from urwid import signals
from . import _raw_display_base, escape
from .common import INPUT_DESCRIPTORS_CHANGED
def _get_gpm_codes(self) -> list[int]:
    codes = []
    try:
        while self.gpm_mev is not None and self.gpm_event_pending:
            codes.extend(self._encode_gpm_event())
    except OSError as e:
        if e.args[0] != 11:
            raise
    return codes