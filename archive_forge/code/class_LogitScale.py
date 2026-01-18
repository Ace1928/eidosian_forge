import inspect
import textwrap
import numpy as np
import matplotlib as mpl
from matplotlib import _api, _docstring
from matplotlib.ticker import (
from matplotlib.transforms import Transform, IdentityTransform
class LogitScale(ScaleBase):
    """
    Logit scale for data between zero and one, both excluded.

    This scale is similar to a log scale close to zero and to one, and almost
    linear around 0.5. It maps the interval ]0, 1[ onto ]-infty, +infty[.
    """
    name = 'logit'

    def __init__(self, axis, nonpositive='mask', *, one_half='\\frac{1}{2}', use_overline=False):
        """
        Parameters
        ----------
        axis : `~matplotlib.axis.Axis`
            Currently unused.
        nonpositive : {'mask', 'clip'}
            Determines the behavior for values beyond the open interval ]0, 1[.
            They can either be masked as invalid, or clipped to a number very
            close to 0 or 1.
        use_overline : bool, default: False
            Indicate the usage of survival notation (\\overline{x}) in place of
            standard notation (1-x) for probability close to one.
        one_half : str, default: r"\\frac{1}{2}"
            The string used for ticks formatter to represent 1/2.
        """
        self._transform = LogitTransform(nonpositive)
        self._use_overline = use_overline
        self._one_half = one_half

    def get_transform(self):
        """Return the `.LogitTransform` associated with this scale."""
        return self._transform

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(LogitLocator())
        axis.set_major_formatter(LogitFormatter(one_half=self._one_half, use_overline=self._use_overline))
        axis.set_minor_locator(LogitLocator(minor=True))
        axis.set_minor_formatter(LogitFormatter(minor=True, one_half=self._one_half, use_overline=self._use_overline))

    def limit_range_for_scale(self, vmin, vmax, minpos):
        """
        Limit the domain to values between 0 and 1 (excluded).
        """
        if not np.isfinite(minpos):
            minpos = 1e-07
        return (minpos if vmin <= 0 else vmin, 1 - minpos if vmax >= 1 else vmax)