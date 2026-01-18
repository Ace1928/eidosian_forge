from __future__ import annotations
import atexit
import builtins
import io
import logging
import math
import os
import re
import struct
import sys
import tempfile
import warnings
from collections.abc import Callable, MutableMapping
from enum import IntEnum
from pathlib import Path
from . import (
from ._binary import i32le, o32be, o32le
from ._util import DeferredError, is_path
class _E:

    def __init__(self, scale, offset):
        self.scale = scale
        self.offset = offset

    def __neg__(self):
        return _E(-self.scale, -self.offset)

    def __add__(self, other):
        if isinstance(other, _E):
            return _E(self.scale + other.scale, self.offset + other.offset)
        return _E(self.scale, self.offset + other)
    __radd__ = __add__

    def __sub__(self, other):
        return self + -other

    def __rsub__(self, other):
        return other + -self

    def __mul__(self, other):
        if isinstance(other, _E):
            return NotImplemented
        return _E(self.scale * other, self.offset * other)
    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, _E):
            return NotImplemented
        return _E(self.scale / other, self.offset / other)