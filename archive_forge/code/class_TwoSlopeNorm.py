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
class TwoSlopeNorm(Normalize):

    def __init__(self, vcenter, vmin=None, vmax=None):
        """
        Normalize data with a set center.

        Useful when mapping data with an unequal rates of change around a
        conceptual center, e.g., data that range from -2 to 4, with 0 as
        the midpoint.

        Parameters
        ----------
        vcenter : float
            The data value that defines ``0.5`` in the normalization.
        vmin : float, optional
            The data value that defines ``0.0`` in the normalization.
            Defaults to the min value of the dataset.
        vmax : float, optional
            The data value that defines ``1.0`` in the normalization.
            Defaults to the max value of the dataset.

        Examples
        --------
        This maps data value -4000 to 0., 0 to 0.5, and +10000 to 1.0; data
        between is linearly interpolated::

            >>> import matplotlib.colors as mcolors
            >>> offset = mcolors.TwoSlopeNorm(vmin=-4000.,
                                              vcenter=0., vmax=10000)
            >>> data = [-4000., -2000., 0., 2500., 5000., 7500., 10000.]
            >>> offset(data)
            array([0., 0.25, 0.5, 0.625, 0.75, 0.875, 1.0])
        """
        super().__init__(vmin=vmin, vmax=vmax)
        self._vcenter = vcenter
        if vcenter is not None and vmax is not None and (vcenter >= vmax):
            raise ValueError('vmin, vcenter, and vmax must be in ascending order')
        if vcenter is not None and vmin is not None and (vcenter <= vmin):
            raise ValueError('vmin, vcenter, and vmax must be in ascending order')

    @property
    def vcenter(self):
        return self._vcenter

    @vcenter.setter
    def vcenter(self, value):
        if value != self._vcenter:
            self._vcenter = value
            self._changed()

    def autoscale_None(self, A):
        """
        Get vmin and vmax.

        If vcenter isn't in the range [vmin, vmax], either vmin or vmax
        is expanded so that vcenter lies in the middle of the modified range
        [vmin, vmax].
        """
        super().autoscale_None(A)
        if self.vmin >= self.vcenter:
            self.vmin = self.vcenter - (self.vmax - self.vcenter)
        if self.vmax <= self.vcenter:
            self.vmax = self.vcenter + (self.vcenter - self.vmin)

    def __call__(self, value, clip=None):
        """
        Map value to the interval [0, 1]. The *clip* argument is unused.
        """
        result, is_scalar = self.process_value(value)
        self.autoscale_None(result)
        if not self.vmin <= self.vcenter <= self.vmax:
            raise ValueError('vmin, vcenter, vmax must increase monotonically')
        result = np.ma.masked_array(np.interp(result, [self.vmin, self.vcenter, self.vmax], [0, 0.5, 1], left=-np.inf, right=np.inf), mask=np.ma.getmask(result))
        if is_scalar:
            result = np.atleast_1d(result)[0]
        return result

    def inverse(self, value):
        if not self.scaled():
            raise ValueError('Not invertible until both vmin and vmax are set')
        (vmin,), _ = self.process_value(self.vmin)
        (vmax,), _ = self.process_value(self.vmax)
        (vcenter,), _ = self.process_value(self.vcenter)
        result = np.interp(value, [0, 0.5, 1], [vmin, vcenter, vmax], left=-np.inf, right=np.inf)
        return result