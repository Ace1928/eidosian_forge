import pyglet.gl as pgl
from sympy.core import S
from sympy.plotting.pygletplot.color_scheme import ColorScheme
from sympy.plotting.pygletplot.plot_mode import PlotMode
from sympy.utilities.iterables import is_sequence
from time import sleep
from threading import Thread, Event, RLock
import warnings
@synchronized
def _set_style(self, v):
    if v is None:
        return
    if v == '':
        step_max = 0
        for i in self.intervals:
            if i.v_steps is None:
                continue
            step_max = max([step_max, int(i.v_steps)])
        v = ['both', 'solid'][step_max > 40]
    if v not in self.styles:
        raise ValueError('v should be there in self.styles')
    if v == self._style:
        return
    self._style = v