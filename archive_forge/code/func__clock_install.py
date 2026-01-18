from math import sqrt, cos, sin, pi
from collections import ChainMap
from kivy.event import EventDispatcher
from kivy.clock import Clock
from kivy.compat import string_types, iterkeys
from kivy.weakproxy import WeakProxy
def _clock_install(self):
    if self._clock_installed:
        return
    self._update_ev = Clock.schedule_interval(self._update, self._step)
    self._clock_installed = True