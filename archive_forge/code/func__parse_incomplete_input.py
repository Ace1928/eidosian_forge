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
def _parse_incomplete_input():
    self._input_timeout = None
    self._partial_codes = []
    self.parse_input(event_loop, callback, codes, wait_for_more=False)