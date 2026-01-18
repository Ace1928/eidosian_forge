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
def _check_layout_engines_compat(self, old, new):
    """
        Helper for set_layout engine

        If the figure has used the old engine and added a colorbar then the
        value of colorbar_gridspec must be the same on the new engine.
        """
    if old is None or new is None:
        return True
    if old.colorbar_gridspec == new.colorbar_gridspec:
        return True
    for ax in self.axes:
        if hasattr(ax, '_colorbar'):
            return False
    return True