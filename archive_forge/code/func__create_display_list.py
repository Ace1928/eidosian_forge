import pyglet.gl as pgl
from sympy.core import S
from sympy.plotting.pygletplot.color_scheme import ColorScheme
from sympy.plotting.pygletplot.plot_mode import PlotMode
from sympy.utilities.iterables import is_sequence
from time import sleep
from threading import Thread, Event, RLock
import warnings
def _create_display_list(self, function):
    dl = pgl.glGenLists(1)
    pgl.glNewList(dl, pgl.GL_COMPILE)
    function()
    pgl.glEndList()
    return dl