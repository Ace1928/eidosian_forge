import logging
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, collections, cm, colors, contour, ticker
import matplotlib.artist as martist
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.spines as mspines
import matplotlib.transforms as mtransforms
from matplotlib import _docstring
def _get_ticker_locator_formatter(self):
    """
        Return the ``locator`` and ``formatter`` of the colorbar.

        If they have not been defined (i.e. are *None*), the formatter and
        locator are retrieved from the axis, or from the value of the
        boundaries for a boundary norm.

        Called by update_ticks...
        """
    locator = self._locator
    formatter = self._formatter
    minorlocator = self._minorlocator
    if isinstance(self.norm, colors.BoundaryNorm):
        b = self.norm.boundaries
        if locator is None:
            locator = ticker.FixedLocator(b, nbins=10)
        if minorlocator is None:
            minorlocator = ticker.FixedLocator(b)
    elif isinstance(self.norm, colors.NoNorm):
        if locator is None:
            nv = len(self._values)
            base = 1 + int(nv / 10)
            locator = ticker.IndexLocator(base=base, offset=0.5)
    elif self.boundaries is not None:
        b = self._boundaries[self._inside]
        if locator is None:
            locator = ticker.FixedLocator(b, nbins=10)
    else:
        if locator is None:
            locator = self._long_axis().get_major_locator()
        if minorlocator is None:
            minorlocator = self._long_axis().get_minor_locator()
    if minorlocator is None:
        minorlocator = ticker.NullLocator()
    if formatter is None:
        formatter = self._long_axis().get_major_formatter()
    self._locator = locator
    self._formatter = formatter
    self._minorlocator = minorlocator
    _log.debug('locator: %r', locator)