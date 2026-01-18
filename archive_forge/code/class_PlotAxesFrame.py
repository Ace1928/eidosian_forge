import pyglet.gl as pgl
from pyglet import font
from sympy.core import S
from sympy.plotting.pygletplot.plot_object import PlotObject
from sympy.plotting.pygletplot.util import billboard_matrix, dot_product, \
from sympy.utilities.iterables import is_sequence
class PlotAxesFrame(PlotAxesBase):

    def __init__(self, parent_axes):
        super().__init__(parent_axes)

    def draw_background(self, color):
        pass

    def draw_axis(self, axis, color):
        raise NotImplementedError()