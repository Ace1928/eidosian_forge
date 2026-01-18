import os
import re
import sys
import time
import codecs
import locale
import select
import struct
import platform
import warnings
import functools
import contextlib
import collections
from .color import COLOR_DISTANCE_ALGORITHMS
from .keyboard import (_time_left,
from .sequences import Termcap, Sequence, SequenceTextWrapper
from .colorspace import RGB_256TABLE
from .formatters import (COLORS,
from ._capabilities import CAPABILITY_DATABASE, CAPABILITIES_ADDITIVES, CAPABILITIES_RAW_MIXIN
def __init__streams(self):
    stream_fd = None
    if self._stream is None:
        self._stream = sys.__stdout__
    if not hasattr(self._stream, 'fileno'):
        self.errors.append('stream has no fileno method')
    elif not callable(self._stream.fileno):
        self.errors.append('stream.fileno is not callable')
    else:
        try:
            stream_fd = self._stream.fileno()
        except ValueError as err:
            self.errors.append('Unable to determine output stream file descriptor: %s' % err)
        else:
            self._is_a_tty = os.isatty(stream_fd)
            if not self._is_a_tty:
                self.errors.append('stream not a TTY')
    if self._stream in (sys.__stdout__, sys.__stderr__):
        try:
            self._keyboard_fd = sys.__stdin__.fileno()
        except (AttributeError, ValueError) as err:
            self.errors.append('Unable to determine input stream file descriptor: %s' % err)
        else:
            if not self.is_a_tty:
                self.errors.append('Output stream is not a TTY')
                self._keyboard_fd = None
            elif not os.isatty(self._keyboard_fd):
                self.errors.append('Input stream is not a TTY')
                self._keyboard_fd = None
    else:
        self.errors.append('Output stream is not a default stream')
    self._init_descriptor = stream_fd
    if stream_fd is None:
        try:
            self._init_descriptor = sys.__stdout__.fileno()
        except ValueError as err:
            self.errors.append('Unable to determine __stdout__ file descriptor: %s' % err)