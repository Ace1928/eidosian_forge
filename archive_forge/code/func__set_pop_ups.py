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
def _set_pop_ups(self, pop_ups: bool) -> None:
    warnings.warn(f'method `{self.__class__.__name__}._set_pop_ups` is deprecated, please use `{self.__class__.__name__}.pop_ups` property', DeprecationWarning, stacklevel=2)
    self.pop_ups = pop_ups