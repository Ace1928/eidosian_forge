from contextlib import ExitStack
import inspect
import itertools
import logging
from numbers import Integral
import threading
import numpy as np
import matplotlib as mpl
from matplotlib import _blocking_input, backend_bases, _docstring, projections
from matplotlib.artist import (
from matplotlib.backend_bases import (
import matplotlib._api as _api
import matplotlib.cbook as cbook
import matplotlib.colorbar as cbar
import matplotlib.image as mimage
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from matplotlib.layout_engine import (
import matplotlib.legend as mlegend
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.transforms import (Affine2D, Bbox, BboxTransformTo,
def _remove_axes(self, ax, owners):
    """
        Common helper for removal of standard axes (via delaxes) and of child axes.

        Parameters
        ----------
        ax : `~.AxesBase`
            The Axes to remove.
        owners
            List of objects (list or _AxesStack) "owning" the axes, from which the Axes
            will be remove()d.
        """
    for owner in owners:
        owner.remove(ax)
    self._axobservers.process('_axes_change_event', self)
    self.stale = True
    self.canvas.release_mouse(ax)
    for name in ax._axis_names:
        grouper = ax._shared_axes[name]
        siblings = [other for other in grouper.get_siblings(ax) if other is not ax]
        if not siblings:
            continue
        grouper.remove(ax)
        remaining_axis = siblings[0]._axis_map[name]
        remaining_axis.get_major_formatter().set_axis(remaining_axis)
        remaining_axis.get_major_locator().set_axis(remaining_axis)
        remaining_axis.get_minor_formatter().set_axis(remaining_axis)
        remaining_axis.get_minor_locator().set_axis(remaining_axis)
    ax._twinned_axes.remove(ax)