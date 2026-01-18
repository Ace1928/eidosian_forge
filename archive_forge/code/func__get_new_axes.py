import functools
import numpy as np
import matplotlib as mpl
from matplotlib import _api
from matplotlib.gridspec import SubplotSpec
import matplotlib.transforms as mtransforms
from . import axes_size as Size
def _get_new_axes(self, *, axes_class=None, **kwargs):
    axes = self._axes
    if axes_class is None:
        axes_class = type(axes)
    return axes_class(axes.get_figure(), axes.get_position(original=True), **kwargs)