from __future__ import annotations
import contextlib
import functools
import logging
import selectors
import socket
import sys
import threading
import typing
from ctypes import byref
from ctypes.wintypes import DWORD
from urwid import signals
from . import _raw_display_base, _win32, escape
from .common import INPUT_DESCRIPTORS_CHANGED
Return the terminal dimensions (num columns, num rows).