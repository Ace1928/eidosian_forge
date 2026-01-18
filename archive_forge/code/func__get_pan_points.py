from collections.abc import Iterable, Sequence
from contextlib import ExitStack
import functools
import inspect
import logging
from numbers import Real
from operator import attrgetter
import types
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, _docstring, offsetbox
import matplotlib.artist as martist
import matplotlib.axis as maxis
from matplotlib.cbook import _OrderedSet, _check_1d, index_of
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.font_manager as font_manager
from matplotlib.gridspec import SubplotSpec
import matplotlib.image as mimage
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.rcsetup import cycler, validate_axisbelow
import matplotlib.spines as mspines
import matplotlib.table as mtable
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
def _get_pan_points(self, button, key, x, y):
    """
        Helper function to return the new points after a pan.

        This helper function returns the points on the axis after a pan has
        occurred. This is a convenience method to abstract the pan logic
        out of the base setter.
        """

    def format_deltas(key, dx, dy):
        if key == 'control':
            if abs(dx) > abs(dy):
                dy = dx
            else:
                dx = dy
        elif key == 'x':
            dy = 0
        elif key == 'y':
            dx = 0
        elif key == 'shift':
            if 2 * abs(dx) < abs(dy):
                dx = 0
            elif 2 * abs(dy) < abs(dx):
                dy = 0
            elif abs(dx) > abs(dy):
                dy = dy / abs(dy) * abs(dx)
            else:
                dx = dx / abs(dx) * abs(dy)
        return (dx, dy)
    p = self._pan_start
    dx = x - p.x
    dy = y - p.y
    if dx == dy == 0:
        return
    if button == 1:
        dx, dy = format_deltas(key, dx, dy)
        result = p.bbox.translated(-dx, -dy).transformed(p.trans_inverse)
    elif button == 3:
        try:
            dx = -dx / self.bbox.width
            dy = -dy / self.bbox.height
            dx, dy = format_deltas(key, dx, dy)
            if self.get_aspect() != 'auto':
                dx = dy = 0.5 * (dx + dy)
            alpha = np.power(10.0, (dx, dy))
            start = np.array([p.x, p.y])
            oldpoints = p.lim.transformed(p.trans)
            newpoints = start + alpha * (oldpoints - start)
            result = mtransforms.Bbox(newpoints).transformed(p.trans_inverse)
        except OverflowError:
            _api.warn_external('Overflow while panning')
            return
    else:
        return
    valid = np.isfinite(result.transformed(p.trans))
    points = result.get_points().astype(object)
    points[~valid] = None
    return points