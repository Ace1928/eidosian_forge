import pyglet.gl as pgl
from sympy.core import S
from sympy.plotting.pygletplot.color_scheme import ColorScheme
from sympy.plotting.pygletplot.plot_mode import PlotMode
from sympy.utilities.iterables import is_sequence
from time import sleep
from threading import Thread, Event, RLock
import warnings
@synchronized
def _set_color(self, v):
    try:
        if v is not None:
            if is_sequence(v):
                v = ColorScheme(*v)
            else:
                v = ColorScheme(v)
        if repr(v) == repr(self._color):
            return
        self._on_change_color(v)
        self._color = v
    except Exception as e:
        raise RuntimeError('Color change failed. Reason: %s' % str(e))