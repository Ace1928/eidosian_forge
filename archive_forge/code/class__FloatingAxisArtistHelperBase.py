from the axis as some gridlines can never pass any axis.
import numpy as np
import matplotlib as mpl
from matplotlib import _api
import matplotlib.axes as maxes
from matplotlib.path import Path
from mpl_toolkits.axes_grid1 import mpl_axes
from .axisline_style import AxislineStyle  # noqa
from .axis_artist import AxisArtist, GridlinesCollection
class _FloatingAxisArtistHelperBase(_AxisArtistHelperBase):

    def __init__(self, nth_coord, value):
        self.nth_coord = nth_coord
        self._value = value
        super().__init__()

    def get_nth_coord(self):
        return self.nth_coord

    def get_line(self, axes):
        raise RuntimeError('get_line method should be defined by the derived class')