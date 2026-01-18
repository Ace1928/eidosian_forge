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
def _test_run_screen_event_loop(self):
    """
        >>> w = _refl("widget")
        >>> scr = _refl("screen")
        >>> scr.get_cols_rows_rval = (10, 5)
        >>> scr.get_input_rval = [], []
        >>> ml = MainLoop(w, screen=scr)
        >>> def stop_now(loop, data):
        ...     raise ExitMainLoop()
        >>> handle = ml.set_alarm_in(0, stop_now)
        >>> try:
        ...     ml._run_screen_event_loop()
        ... except ExitMainLoop:
        ...     pass
        screen.get_cols_rows()
        widget.render((10, 5), focus=True)
        screen.draw_screen((10, 5), None)
        screen.set_input_timeouts(0.0)
        screen.get_input(True)
        """