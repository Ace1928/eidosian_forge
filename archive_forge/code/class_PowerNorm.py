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
class PowerNorm(Normalize):
    """
    Linearly map a given value to the 0-1 range and then apply
    a power-law normalization over that range.

    Parameters
    ----------
    gamma : float
        Power law exponent.
    vmin, vmax : float or None
        If *vmin* and/or *vmax* is not given, they are initialized from the
        minimum and maximum value, respectively, of the first input
        processed; i.e., ``__call__(A)`` calls ``autoscale_None(A)``.
    clip : bool, default: False
        Determines the behavior for mapping values outside the range
        ``[vmin, vmax]``.

        If clipping is off, values outside the range ``[vmin, vmax]`` are also
        transformed by the power function, resulting in values outside ``[0, 1]``. For
        a standard use with colormaps, this behavior is desired because colormaps
        mark these outside values with specific colors for *over* or *under*.

        If ``True`` values falling outside the range ``[vmin, vmax]``,
        are mapped to 0 or 1, whichever is closer. This makes these values
        indistinguishable from regular boundary values and can lead to
        misinterpretation of the data.

    Notes
    -----
    The normalization formula is

    .. math::

        \\left ( \\frac{x - v_{min}}{v_{max}  - v_{min}} \\right )^{\\gamma}
    """

    def __init__(self, gamma, vmin=None, vmax=None, clip=False):
        super().__init__(vmin, vmax, clip)
        self.gamma = gamma

    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip
        result, is_scalar = self.process_value(value)
        self.autoscale_None(result)
        gamma = self.gamma
        vmin, vmax = (self.vmin, self.vmax)
        if vmin > vmax:
            raise ValueError('minvalue must be less than or equal to maxvalue')
        elif vmin == vmax:
            result.fill(0)
        else:
            if clip:
                mask = np.ma.getmask(result)
                result = np.ma.array(np.clip(result.filled(vmax), vmin, vmax), mask=mask)
            resdat = result.data
            resdat -= vmin
            resdat[resdat < 0] = 0
            np.power(resdat, gamma, resdat)
            resdat /= (vmax - vmin) ** gamma
            result = np.ma.array(resdat, mask=result.mask, copy=False)
        if is_scalar:
            result = result[0]
        return result

    def inverse(self, value):
        if not self.scaled():
            raise ValueError('Not invertible until scaled')
        gamma = self.gamma
        vmin, vmax = (self.vmin, self.vmax)
        if np.iterable(value):
            val = np.ma.asarray(value)
            return np.ma.power(val, 1.0 / gamma) * (vmax - vmin) + vmin
        else:
            return pow(value, 1.0 / gamma) * (vmax - vmin) + vmin