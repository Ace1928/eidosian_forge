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
def _on_update_palette_entry(self, name: str | None, *attrspecs: AttrSpec):
    a: AttrSpec = attrspecs[{16: 0, 1: 1, 88: 2, 256: 3, 2 ** 24: 4}[self.colors]]
    self._pal_attrspec[name] = a
    self._pal_escape[name] = self._attrspec_to_escape(a)