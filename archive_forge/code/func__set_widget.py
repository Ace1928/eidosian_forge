from __future__ import annotations
import heapq
import logging
import os
import sys
import time
import typing
import warnings
from contextlib import suppress
from urwid import display, signals
from urwid.command_map import Command, command_map
from urwid.display.common import INPUT_DESCRIPTORS_CHANGED
from urwid.util import StoppingContext, is_mouse_event
from urwid.widget import PopUpTarget
from .abstract_loop import ExitMainLoop
from .select_loop import SelectEventLoop
def _set_widget(self, widget: Widget) -> None:
    warnings.warn(f'method `{self.__class__.__name__}._set_widget` is deprecated, please use `{self.__class__.__name__}.widget` property', DeprecationWarning, stacklevel=2)
    self.widget = widget