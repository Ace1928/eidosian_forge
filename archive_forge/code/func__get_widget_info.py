from __future__ import annotations
import contextlib
import dataclasses
import typing
import warnings
import weakref
from contextlib import suppress
from urwid.str_util import calc_text_pos, calc_width
from urwid.text_layout import LayoutSegment, trim_line
from urwid.util import (
def _get_widget_info(self):
    warnings.warn(f'Method `{self.__class__.__name__}._get_widget_info` is deprecated, please use property `{self.__class__.__name__}.widget_info`', DeprecationWarning, stacklevel=2)
    return self.widget_info