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
def __init__keycodes(self):
    self._keycodes = get_keyboard_codes()
    for key_code, key_name in self._keycodes.items():
        setattr(self, key_name, key_code)
    self._keymap = get_keyboard_sequences(self)
    self._keymap_prefixes = get_leading_prefixes(self._keymap)
    self._keyboard_buf = collections.deque()
    if self._keyboard_fd is not None:
        if IS_WINDOWS:
            self._encoding = get_console_input_encoding() or locale.getpreferredencoding() or 'UTF-8'
        else:
            self._encoding = locale.getpreferredencoding() or 'UTF-8'
        try:
            self._keyboard_decoder = codecs.getincrementaldecoder(self._encoding)()
        except LookupError as err:
            warnings.warn('LookupError: {0}, defaulting to UTF-8 for keyboard.'.format(err))
            self._encoding = 'UTF-8'
            self._keyboard_decoder = codecs.getincrementaldecoder(self._encoding)()