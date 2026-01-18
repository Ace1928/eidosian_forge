from __future__ import annotations
import os
import sys
from abc import abstractmethod
from asyncio import get_running_loop
from contextlib import contextmanager
from ..utils import SPHINX_AUTODOC_RUNNING
from ctypes import Array, pointer
from ctypes.wintypes import DWORD, HANDLE
from typing import Callable, ContextManager, Iterable, Iterator, TextIO
from prompt_toolkit.eventloop import run_in_executor_with_context
from prompt_toolkit.eventloop.win32 import create_win32_event, wait_for_handles
from prompt_toolkit.key_binding.key_processor import KeyPress
from prompt_toolkit.keys import Keys
from prompt_toolkit.mouse_events import MouseButton, MouseEventType
from prompt_toolkit.win32_types import (
from .ansi_escape_sequences import REVERSE_ANSI_SEQUENCES
from .base import Input
def _insert_key_data(self, key_press: KeyPress) -> KeyPress:
    """
        Insert KeyPress data, for vt100 compatibility.
        """
    if key_press.data:
        return key_press
    if isinstance(key_press.key, Keys):
        data = REVERSE_ANSI_SEQUENCES.get(key_press.key, '')
    else:
        data = ''
    return KeyPress(key_press.key, data)