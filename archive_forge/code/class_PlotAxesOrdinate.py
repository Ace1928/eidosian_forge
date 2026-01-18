import pyglet.gl as pgl
from pyglet import font
from sympy.core import S
from sympy.plotting.pygletplot.plot_object import PlotObject
from sympy.plotting.pygletplot.util import billboard_matrix, dot_product, \
from sympy.utilities.iterables import is_sequence
class PlotAxesOrdinate(PlotAxesBase):

    def __init__(self, parent_axes):
        super().__init__(parent_axes)

    def draw_axis(self, axis, color):
        ticks = self._p._axis_ticks[axis]
        radius = self._p._tick_length / 2.0
        if len(ticks) < 2:
            return
        axis_lines = [[0, 0, 0], [0, 0, 0]]
        axis_lines[0][axis], axis_lines[1][axis] = (ticks[0], ticks[-1])
        axis_vector = vec_sub(axis_lines[1], axis_lines[0])
        pos_z = get_direction_vectors()[2]
        d = abs(dot_product(axis_vector, pos_z))
        d = d / vec_mag(axis_vector)
        labels_visible = abs(d - 1.0) > 0.02
        for tick in ticks:
            self.draw_tick_line(axis, color, radius, tick, labels_visible)
        self.draw_axis_line(axis, color, ticks[0], ticks[-1], labels_visible)

    def draw_axis_line(self, axis, color, a_min, a_max, labels_visible):
        axis_line = [[0, 0, 0], [0, 0, 0]]
        axis_line[0][axis], axis_line[1][axis] = (a_min, a_max)
        self.draw_line(axis_line, color)
        if labels_visible:
            self.draw_axis_line_labels(axis, color, axis_line)

    def draw_axis_line_labels(self, axis, color, axis_line):
        if not self._p._label_axes:
            return
        axis_labels = [axis_line[0][:], axis_line[1][:]]
        axis_labels[0][axis] -= 0.3
        axis_labels[1][axis] += 0.3
        a_str = ['X', 'Y', 'Z'][axis]
        self.draw_text('-' + a_str, axis_labels[0], color)
        self.draw_text('+' + a_str, axis_labels[1], color)

    def draw_tick_line(self, axis, color, radius, tick, labels_visible):
        tick_axis = {0: 1, 1: 0, 2: 1}[axis]
        tick_line = [[0, 0, 0], [0, 0, 0]]
        tick_line[0][axis] = tick_line[1][axis] = tick
        tick_line[0][tick_axis], tick_line[1][tick_axis] = (-radius, radius)
        self.draw_line(tick_line, color)
        if labels_visible:
            self.draw_tick_line_label(axis, color, radius, tick)

    def draw_tick_line_label(self, axis, color, radius, tick):
        if not self._p._label_axes:
            return
        tick_label_vector = [0, 0, 0]
        tick_label_vector[axis] = tick
        tick_label_vector[{0: 1, 1: 0, 2: 1}[axis]] = [-1, 1, 1][axis] * radius * 3.5
        self.draw_text(str(tick), tick_label_vector, color, scale=0.5)