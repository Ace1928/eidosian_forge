from the axis as some gridlines can never pass any axis.
import numpy as np
import matplotlib as mpl
from matplotlib import _api
import matplotlib.axes as maxes
from matplotlib.path import Path
from mpl_toolkits.axes_grid1 import mpl_axes
from .axisline_style import AxislineStyle  # noqa
from .axis_artist import AxisArtist, GridlinesCollection
class FloatingAxisArtistHelperRectilinear(_FloatingAxisArtistHelperBase):

    def __init__(self, axes, nth_coord, passingthrough_point, axis_direction='bottom'):
        super().__init__(nth_coord, passingthrough_point)
        self._axis_direction = axis_direction
        self.axis = [axes.xaxis, axes.yaxis][self.nth_coord]

    def get_line(self, axes):
        fixed_coord = 1 - self.nth_coord
        data_to_axes = axes.transData - axes.transAxes
        p = data_to_axes.transform([self._value, self._value])
        return Path(self._to_xy((0, 1), const=p[fixed_coord]))

    def get_line_transform(self, axes):
        return axes.transAxes

    def get_axislabel_transform(self, axes):
        return axes.transAxes

    def get_axislabel_pos_angle(self, axes):
        """
        Return the label reference position in transAxes.

        get_label_transform() returns a transform of (transAxes+offset)
        """
        angle = [0, 90][self.nth_coord]
        fixed_coord = 1 - self.nth_coord
        data_to_axes = axes.transData - axes.transAxes
        p = data_to_axes.transform([self._value, self._value])
        verts = self._to_xy(0.5, const=p[fixed_coord])
        if 0 <= verts[fixed_coord] <= 1:
            return (verts, angle)
        else:
            return (None, None)

    def get_tick_transform(self, axes):
        return axes.transData

    def get_tick_iterators(self, axes):
        """tick_loc, tick_angle, tick_label"""
        if self.nth_coord == 0:
            angle_normal, angle_tangent = (90, 0)
        else:
            angle_normal, angle_tangent = (0, 90)
        major = self.axis.major
        major_locs = major.locator()
        major_labels = major.formatter.format_ticks(major_locs)
        minor = self.axis.minor
        minor_locs = minor.locator()
        minor_labels = minor.formatter.format_ticks(minor_locs)
        data_to_axes = axes.transData - axes.transAxes

        def _f(locs, labels):
            for loc, label in zip(locs, labels):
                c = self._to_xy(loc, const=self._value)
                c1, c2 = data_to_axes.transform(c)
                if 0 <= c1 <= 1 and 0 <= c2 <= 1:
                    yield (c, angle_normal, angle_tangent, label)
        return (_f(major_locs, major_labels), _f(minor_locs, minor_labels))