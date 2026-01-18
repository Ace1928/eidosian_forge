import copy
import logging
from numbers import Integral
import numpy as np
import matplotlib as mpl
from matplotlib import _api, _pylab_helpers, _tight_layout
from matplotlib.transforms import Bbox
@staticmethod
def _check_gridspec_exists(figure, nrows, ncols):
    """
        Check if the figure already has a gridspec with these dimensions,
        or create a new one
        """
    for ax in figure.get_axes():
        gs = ax.get_gridspec()
        if gs is not None:
            if hasattr(gs, 'get_topmost_subplotspec'):
                gs = gs.get_topmost_subplotspec().get_gridspec()
            if gs.get_geometry() == (nrows, ncols):
                return gs
    return GridSpec(nrows, ncols, figure=figure)