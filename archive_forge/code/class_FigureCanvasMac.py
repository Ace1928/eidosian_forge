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
class FigureCanvasMac(FigureCanvasAgg, _macosx.FigureCanvas, FigureCanvasBase):
    required_interactive_framework = 'macosx'
    _timer_cls = TimerMac
    manager_class = _api.classproperty(lambda cls: FigureManagerMac)

    def __init__(self, figure):
        super().__init__(figure=figure)
        self._draw_pending = False
        self._is_drawing = False
        self._timers = set()

    def draw(self):
        """Render the figure and update the macosx canvas."""
        if self._is_drawing:
            return
        with cbook._setattr_cm(self, _is_drawing=True):
            super().draw()
        self.update()

    def draw_idle(self):
        if not (getattr(self, '_draw_pending', False) or getattr(self, '_is_drawing', False)):
            self._draw_pending = True
            self._single_shot_timer(self._draw_idle)

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

    def _draw_idle(self):
        """
        Draw method for singleshot timer

        This draw method can be added to a singleshot timer, which can
        accumulate draws while the eventloop is spinning. This method will
        then only draw the first time and short-circuit the others.
        """
        with self._idle_draw_cntx():
            if not self._draw_pending:
                return
            self._draw_pending = False
            self.draw()

    def blit(self, bbox=None):
        super().blit(bbox)
        self.update()

    def resize(self, width, height):
        scale = self.figure.dpi / self.device_pixel_ratio
        width /= scale
        height /= scale
        self.figure.set_size_inches(width, height, forward=False)
        ResizeEvent('resize_event', self)._process()
        self.draw_idle()

    def start_event_loop(self, timeout=0):
        with _maybe_allow_interrupt():
            self._start_event_loop(timeout=timeout)