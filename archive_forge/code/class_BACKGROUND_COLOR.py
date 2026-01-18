from __future__ import annotations
import sys
import os
from ctypes import ArgumentError, byref, c_char, c_long, c_uint, c_ulong, pointer
from ctypes.wintypes import DWORD, HANDLE
from typing import Callable, TextIO, TypeVar
from prompt_toolkit.cursor_shapes import CursorShape
from prompt_toolkit.data_structures import Size
from prompt_toolkit.styles import ANSI_COLOR_NAMES, Attrs
from prompt_toolkit.utils import get_cwidth
from prompt_toolkit.win32_types import (
from ..utils import SPHINX_AUTODOC_RUNNING
from .base import Output
from .color_depth import ColorDepth
class BACKGROUND_COLOR:
    BLACK = 0
    BLUE = 16
    GREEN = 32
    CYAN = 48
    RED = 64
    MAGENTA = 80
    YELLOW = 96
    GRAY = 112
    INTENSITY = 128