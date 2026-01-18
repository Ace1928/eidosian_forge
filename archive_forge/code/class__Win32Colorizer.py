from __future__ import annotations
import os
import sys
import time
import unittest as pyunit
import warnings
from collections import OrderedDict
from types import TracebackType
from typing import TYPE_CHECKING, List, Optional, Tuple, Type, Union
from zope.interface import implementer
from typing_extensions import TypeAlias
from twisted.python import log, reflect
from twisted.python.components import proxyForInterface
from twisted.python.failure import Failure
from twisted.python.util import untilConcludes
from twisted.trial import itrial, util
class _Win32Colorizer:
    """
    See _AnsiColorizer docstring.
    """

    def __init__(self, stream):
        from win32console import FOREGROUND_BLUE, FOREGROUND_GREEN, FOREGROUND_INTENSITY, FOREGROUND_RED, STD_OUTPUT_HANDLE, GetStdHandle
        red, green, blue, bold = (FOREGROUND_RED, FOREGROUND_GREEN, FOREGROUND_BLUE, FOREGROUND_INTENSITY)
        self.stream = stream
        self.screenBuffer = GetStdHandle(STD_OUTPUT_HANDLE)
        self._colors = {'normal': red | green | blue, 'red': red | bold, 'green': green | bold, 'blue': blue | bold, 'yellow': red | green | bold, 'magenta': red | blue | bold, 'cyan': green | blue | bold, 'white': red | green | blue | bold}

    @classmethod
    def supported(cls, stream=sys.stdout):
        try:
            import win32console
            screenBuffer = win32console.GetStdHandle(win32console.STD_OUTPUT_HANDLE)
        except ImportError:
            return False
        import pywintypes
        try:
            screenBuffer.SetConsoleTextAttribute(win32console.FOREGROUND_RED | win32console.FOREGROUND_GREEN | win32console.FOREGROUND_BLUE)
        except pywintypes.error:
            return False
        else:
            return True

    def write(self, text, color):
        color = self._colors[color]
        self.screenBuffer.SetConsoleTextAttribute(color)
        self.stream.write(text)
        self.screenBuffer.SetConsoleTextAttribute(self._colors['normal'])