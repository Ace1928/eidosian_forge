import functools
import itertools
import logging
import math
from numbers import Integral, Number, Real
import numpy as np
from numpy import ma
import matplotlib as mpl
import matplotlib.category  # Register category unit converter as side effect.
import matplotlib.cbook as cbook
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.contour as mcontour
import matplotlib.dates  # noqa # Register date unit converter as side effect.
import matplotlib.image as mimage
import matplotlib.legend as mlegend
import matplotlib.lines as mlines
import matplotlib.markers as mmarkers
import matplotlib.mlab as mlab
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.quiver as mquiver
import matplotlib.stackplot as mstack
import matplotlib.streamplot as mstream
import matplotlib.table as mtable
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import matplotlib.tri as mtri
import matplotlib.units as munits
from matplotlib import _api, _docstring, _preprocess_data
from matplotlib.axes._base import (
from matplotlib.axes._secondary_axes import SecondaryAxis
from matplotlib.container import BarContainer, ErrorbarContainer, StemContainer
@staticmethod
def _errorevery_to_mask(x, errorevery):
    """
        Normalize `errorbar`'s *errorevery* to be a boolean mask for data *x*.

        This function is split out to be usable both by 2D and 3D errorbars.
        """
    if isinstance(errorevery, Integral):
        errorevery = (0, errorevery)
    if isinstance(errorevery, tuple):
        if len(errorevery) == 2 and isinstance(errorevery[0], Integral) and isinstance(errorevery[1], Integral):
            errorevery = slice(errorevery[0], None, errorevery[1])
        else:
            raise ValueError(f'errorevery={errorevery!r} is a not a tuple of two integers')
    elif isinstance(errorevery, slice):
        pass
    elif not isinstance(errorevery, str) and np.iterable(errorevery):
        try:
            x[errorevery]
        except (ValueError, IndexError) as err:
            raise ValueError(f"errorevery={errorevery!r} is iterable but not a valid NumPy fancy index to match 'xerr'/'yerr'") from err
    else:
        raise ValueError(f'errorevery={errorevery!r} is not a recognized value')
    everymask = np.zeros(len(x), bool)
    everymask[errorevery] = True
    return everymask