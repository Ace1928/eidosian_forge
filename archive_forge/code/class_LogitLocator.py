import itertools
import logging
import locale
import math
from numbers import Integral
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
from matplotlib import transforms as mtransforms
class LogitLocator(MaxNLocator):
    """
    Determine the tick locations for logit axes
    """

    def __init__(self, minor=False, *, nbins='auto'):
        """
        Place ticks on the logit locations

        Parameters
        ----------
        nbins : int or 'auto', optional
            Number of ticks. Only used if minor is False.
        minor : bool, default: False
            Indicate if this locator is for minor ticks or not.
        """
        self._minor = minor
        super().__init__(nbins=nbins, steps=[1, 2, 5, 10])

    def set_params(self, minor=None, **kwargs):
        """Set parameters within this locator."""
        if minor is not None:
            self._minor = minor
        super().set_params(**kwargs)

    @property
    def minor(self):
        return self._minor

    @minor.setter
    def minor(self, value):
        self.set_params(minor=value)

    def tick_values(self, vmin, vmax):
        if hasattr(self.axis, 'axes') and self.axis.axes.name == 'polar':
            raise NotImplementedError('Polar axis cannot be logit scaled yet')
        if self._nbins == 'auto':
            if self.axis is not None:
                nbins = self.axis.get_tick_space()
                if nbins < 2:
                    nbins = 2
            else:
                nbins = 9
        else:
            nbins = self._nbins

        def ideal_ticks(x):
            return 10 ** x if x < 0 else 1 - 10 ** (-x) if x > 0 else 0.5
        vmin, vmax = self.nonsingular(vmin, vmax)
        binf = int(np.floor(np.log10(vmin)) if vmin < 0.5 else 0 if vmin < 0.9 else -np.ceil(np.log10(1 - vmin)))
        bsup = int(np.ceil(np.log10(vmax)) if vmax <= 0.5 else 1 if vmax <= 0.9 else -np.floor(np.log10(1 - vmax)))
        numideal = bsup - binf - 1
        if numideal >= 2:
            if numideal > nbins:
                subsampling_factor = math.ceil(numideal / nbins)
                if self._minor:
                    ticklocs = [ideal_ticks(b) for b in range(binf, bsup + 1) if b % subsampling_factor != 0]
                else:
                    ticklocs = [ideal_ticks(b) for b in range(binf, bsup + 1) if b % subsampling_factor == 0]
                return self.raise_if_exceeds(np.array(ticklocs))
            if self._minor:
                ticklocs = []
                for b in range(binf, bsup):
                    if b < -1:
                        ticklocs.extend(np.arange(2, 10) * 10 ** b)
                    elif b == -1:
                        ticklocs.extend(np.arange(2, 5) / 10)
                    elif b == 0:
                        ticklocs.extend(np.arange(6, 9) / 10)
                    else:
                        ticklocs.extend(1 - np.arange(2, 10)[::-1] * 10 ** (-b - 1))
                return self.raise_if_exceeds(np.array(ticklocs))
            ticklocs = [ideal_ticks(b) for b in range(binf, bsup + 1)]
            return self.raise_if_exceeds(np.array(ticklocs))
        if self._minor:
            return []
        return super().tick_values(vmin, vmax)

    def nonsingular(self, vmin, vmax):
        standard_minpos = 1e-07
        initial_range = (standard_minpos, 1 - standard_minpos)
        if vmin > vmax:
            vmin, vmax = (vmax, vmin)
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            vmin, vmax = initial_range
        elif vmax <= 0 or vmin >= 1:
            _api.warn_external('Data has no values between 0 and 1, and therefore cannot be logit-scaled.')
            vmin, vmax = initial_range
        else:
            minpos = self.axis.get_minpos() if self.axis is not None else standard_minpos
            if not np.isfinite(minpos):
                minpos = standard_minpos
            if vmin <= 0:
                vmin = minpos
            if vmax >= 1:
                vmax = 1 - minpos
            if vmin == vmax:
                vmin, vmax = (0.1 * vmin, 1 - 0.1 * vmin)
        return (vmin, vmax)