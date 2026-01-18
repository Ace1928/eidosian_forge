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

        entries - list of (index, red, green, blue) tuples.

        Attempt to set part of the terminal palette (this does not work
        on all terminals.)  The changes are sent as a single escape
        sequence so they should all take effect at the same time.

        0 <= index < 256 (some terminals will only have 16 or 88 colors)
        0 <= red, green, blue < 256
        