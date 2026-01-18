import base64
from collections.abc import Sized, Sequence, Mapping
import functools
import importlib
import inspect
import io
import itertools
from numbers import Real
import re
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import matplotlib as mpl
import numpy as np
from matplotlib import _api, _cm, cbook, scale
from ._color_data import BASE_COLORS, TABLEAU_COLORS, CSS4_COLORS, XKCD_COLORS
def _create_lookup_table(N, data, gamma=1.0):
    """
    Create an *N* -element 1D lookup table.

    This assumes a mapping :math:`f : [0, 1] \\rightarrow [0, 1]`. The returned
    data is an array of N values :math:`y = f(x)` where x is sampled from
    [0, 1].

    By default (*gamma* = 1) x is equidistantly sampled from [0, 1]. The
    *gamma* correction factor :math:`\\gamma` distorts this equidistant
    sampling by :math:`x \\rightarrow x^\\gamma`.

    Parameters
    ----------
    N : int
        The number of elements of the created lookup table; at least 1.

    data : (M, 3) array-like or callable
        Defines the mapping :math:`f`.

        If a (M, 3) array-like, the rows define values (x, y0, y1).  The x
        values must start with x=0, end with x=1, and all x values be in
        increasing order.

        A value between :math:`x_i` and :math:`x_{i+1}` is mapped to the range
        :math:`y^1_{i-1} \\ldots y^0_i` by linear interpolation.

        For the simple case of a y-continuous mapping, y0 and y1 are identical.

        The two values of y are to allow for discontinuous mapping functions.
        E.g. a sawtooth with a period of 0.2 and an amplitude of 1 would be::

            [(0, 1, 0), (0.2, 1, 0), (0.4, 1, 0), ..., [(1, 1, 0)]

        In the special case of ``N == 1``, by convention the returned value
        is y0 for x == 1.

        If *data* is a callable, it must accept and return numpy arrays::

           data(x : ndarray) -> ndarray

        and map values between 0 - 1 to 0 - 1.

    gamma : float
        Gamma correction factor for input distribution x of the mapping.

        See also https://en.wikipedia.org/wiki/Gamma_correction.

    Returns
    -------
    array
        The lookup table where ``lut[x * (N-1)]`` gives the closest value
        for values of x between 0 and 1.

    Notes
    -----
    This function is internally used for `.LinearSegmentedColormap`.
    """
    if callable(data):
        xind = np.linspace(0, 1, N) ** gamma
        lut = np.clip(np.array(data(xind), dtype=float), 0, 1)
        return lut
    try:
        adata = np.array(data)
    except Exception as err:
        raise TypeError('data must be convertible to an array') from err
    _api.check_shape((None, 3), data=adata)
    x = adata[:, 0]
    y0 = adata[:, 1]
    y1 = adata[:, 2]
    if x[0] != 0.0 or x[-1] != 1.0:
        raise ValueError('data mapping points must start with x=0 and end with x=1')
    if (np.diff(x) < 0).any():
        raise ValueError('data mapping points must have x in increasing order')
    if N == 1:
        lut = np.array(y0[-1])
    else:
        x = x * (N - 1)
        xind = (N - 1) * np.linspace(0, 1, N) ** gamma
        ind = np.searchsorted(x, xind)[1:-1]
        distance = (xind[1:-1] - x[ind - 1]) / (x[ind] - x[ind - 1])
        lut = np.concatenate([[y1[0]], distance * (y0[ind] - y1[ind - 1]) + y1[ind - 1], [y0[-1]]])
    return np.clip(lut, 0.0, 1.0)