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
def _run_screen_event_loop(self) -> None:
    """
        This method is used when the screen does not support using external event loops.

        The alarms stored in the SelectEventLoop in :attr:`event_loop` are modified by this method.
        """
    self.logger.debug(f'Starting screen {self.screen!r} event loop')
    next_alarm = None
    while True:
        self.draw_screen()
        if not next_alarm and self.event_loop._alarms:
            next_alarm = heapq.heappop(self.event_loop._alarms)
        keys: list[str] = []
        raw: list[int] = []
        while not keys:
            if next_alarm:
                sec = max(0.0, next_alarm[0] - time.time())
                self.screen.set_input_timeouts(sec)
            else:
                self.screen.set_input_timeouts(None)
            keys, raw = self.screen.get_input(True)
            if not keys and next_alarm:
                sec = next_alarm[0] - time.time()
                if sec <= 0:
                    break
        keys = self.input_filter(keys, raw)
        if keys:
            self.process_input(keys)
        while next_alarm:
            sec = next_alarm[0] - time.time()
            if sec > 0:
                break
            _tm, _tie_break, callback = next_alarm
            callback()
            if self.event_loop._alarms:
                next_alarm = heapq.heappop(self.event_loop._alarms)
            else:
                next_alarm = None
        if 'window resize' in keys:
            self.screen_size = None