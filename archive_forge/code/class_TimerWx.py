import functools
import logging
import math
import pathlib
import sys
import weakref
import numpy as np
import PIL.Image
import matplotlib as mpl
from matplotlib.backend_bases import (
from matplotlib import _api, cbook, backend_tools
from matplotlib._pylab_helpers import Gcf
from matplotlib.path import Path
from matplotlib.transforms import Affine2D
import wx
class TimerWx(TimerBase):
    """Subclass of `.TimerBase` using wx.Timer events."""

    def __init__(self, *args, **kwargs):
        self._timer = wx.Timer()
        self._timer.Notify = self._on_timer
        super().__init__(*args, **kwargs)

    def _timer_start(self):
        self._timer.Start(self._interval, self._single)

    def _timer_stop(self):
        self._timer.Stop()

    def _timer_set_interval(self):
        if self._timer.IsRunning():
            self._timer_start()