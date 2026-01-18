from __future__ import annotations
import abc
import logging
import os
import sys
import typing
import warnings
from urwid import signals
from urwid.util import StoppingContext, int_scale
def _background(self) -> str:
    warnings.warn(f'Method `{self.__class__.__name__}._background` is deprecated, please use property `{self.__class__.__name__}.background`', DeprecationWarning, stacklevel=2)
    return self.background