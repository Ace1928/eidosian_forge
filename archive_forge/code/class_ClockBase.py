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
class ClockBase(ClockBaseBehavior, CyClockBase):
    """The ``default`` kivy clock. See module for details.
    """
    _sleep_obj = None

    def __init__(self, **kwargs):
        super(ClockBase, self).__init__(**kwargs)
        self._sleep_obj = _get_sleep_obj()

    def usleep(self, microseconds):
        _usleep(microseconds, self._sleep_obj)