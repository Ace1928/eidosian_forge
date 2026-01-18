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
def _make_array(inp):
    """
            Convert input into 2D array

            We need to have this internal function rather than
            ``np.asarray(..., dtype=object)`` so that a list of lists
            of lists does not get converted to an array of dimension > 2.

            Returns
            -------
            2D object array
            """
    r0, *rest = inp
    if isinstance(r0, str):
        raise ValueError('List mosaic specification must be 2D')
    for j, r in enumerate(rest, start=1):
        if isinstance(r, str):
            raise ValueError('List mosaic specification must be 2D')
        if len(r0) != len(r):
            raise ValueError(f'All of the rows must be the same length, however the first row ({r0!r}) has length {len(r0)} and row {j} ({r!r}) has length {len(r)}.')
    out = np.zeros((len(inp), len(r0)), dtype=object)
    for j, r in enumerate(inp):
        for k, v in enumerate(r):
            out[j, k] = v
    return out