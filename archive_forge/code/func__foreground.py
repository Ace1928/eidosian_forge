from __future__ import annotations
import abc
import logging
import os
import sys
import typing
import warnings
from urwid import signals
from urwid.util import StoppingContext, int_scale
def _foreground(self) -> str:
    warnings.warn(f'Method `{self.__class__.__name__}._foreground` is deprecated, please use property `{self.__class__.__name__}.foreground`', DeprecationWarning, stacklevel=2)
    return self.foreground