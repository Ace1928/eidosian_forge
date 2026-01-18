from 1.0.5, you can use a timeout of -1::
from sys import platform
from os import environ
from functools import wraps, partial
from kivy.context import register_context
from kivy.config import Config
from kivy.logger import Logger
from kivy.compat import clock as _default_time
import time
from threading import Event as ThreadingEvent
def _check_ready(self, fps, min_sleep, undershoot, event):
    if event.is_set():
        return (True, 0)
    t = self._get_min_timeout_func()
    if not t:
        return (True, 0)
    if not self.interupt_next_only:
        curr_t = self.time()
        sleeptime = min(1 / fps - (curr_t - self._last_tick), t - curr_t)
    else:
        sleeptime = 1 / fps - (self.time() - self._last_tick)
    return (sleeptime - undershoot <= min_sleep, sleeptime - undershoot)