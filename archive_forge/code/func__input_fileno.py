from __future__ import annotations
import abc
import contextlib
import functools
import os
import platform
import selectors
import signal
import socket
import sys
import typing
from urwid import signals, str_util, util
from . import escape
from .common import UNPRINTABLE_TRANS_TABLE, UPDATE_PALETTE_ENTRY, AttrSpec, BaseScreen, RealTerminal
def _input_fileno(self) -> int | None:
    """Returns the fileno of the input stream, or None if it doesn't have one.

        A stream without a fileno can't participate in whatever.
        """
    if hasattr(self._term_input_file, 'fileno'):
        return self._term_input_file.fileno()
    return None