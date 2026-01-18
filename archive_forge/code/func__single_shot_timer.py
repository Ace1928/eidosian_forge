import contextlib
import os
import signal
import socket
import matplotlib as mpl
from matplotlib import _api, cbook
from matplotlib._pylab_helpers import Gcf
from . import _macosx
from .backend_agg import FigureCanvasAgg
from matplotlib.backend_bases import (
def _single_shot_timer(self, callback):
    """Add a single shot timer with the given callback"""

    def callback_func(callback, timer):
        callback()
        self._timers.remove(timer)
        timer.stop()
    timer = self.new_timer(interval=0)
    timer.single_shot = True
    timer.add_callback(callback_func, callback, timer)
    self._timers.add(timer)
    timer.start()