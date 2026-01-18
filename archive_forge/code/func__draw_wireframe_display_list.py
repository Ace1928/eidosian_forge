import pyglet.gl as pgl
from sympy.core import S
from sympy.plotting.pygletplot.color_scheme import ColorScheme
from sympy.plotting.pygletplot.plot_mode import PlotMode
from sympy.utilities.iterables import is_sequence
from time import sleep
from threading import Thread, Event, RLock
import warnings
def _draw_wireframe_display_list(self, dl):
    pgl.glPushAttrib(pgl.GL_ENABLE_BIT | pgl.GL_POLYGON_BIT)
    pgl.glPolygonMode(pgl.GL_FRONT_AND_BACK, pgl.GL_LINE)
    pgl.glEnable(pgl.GL_POLYGON_OFFSET_LINE)
    pgl.glPolygonOffset(-0.005, -50.0)
    pgl.glCallList(dl)
    pgl.glPopAttrib()