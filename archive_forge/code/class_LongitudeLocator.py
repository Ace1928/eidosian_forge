import matplotlib as mpl
from matplotlib.ticker import Formatter, MaxNLocator
import numpy as np
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
class LongitudeLocator(MaxNLocator):
    """
    A locator for longitudes that works even at very small scale.

    Parameters
    ----------
    dms: bool
        Allow the locator to stop on minutes and seconds (False by default)
    """

    def __init__(self, nbins=8, *, dms=False, **kwargs):
        super().__init__(nbins=nbins, dms=dms, **kwargs)

    def set_params(self, **kwargs):
        """Set parameters within this locator."""
        if 'dms' in kwargs:
            self._dms = kwargs.pop('dms')
        MaxNLocator.set_params(self, **kwargs)

    def _guess_steps(self, vmin, vmax):
        dv = abs(vmax - vmin)
        if dv > 180:
            dv -= 180
        if dv > 50.0:
            steps = np.array([1, 2, 3, 6, 10])
        elif not self._dms or dv > 3.0:
            steps = np.array([1, 1.5, 2, 2.5, 3, 5, 10])
        else:
            steps = np.array([1, 10 / 6.0, 15 / 6.0, 20 / 6.0, 30 / 6.0, 10])
        self.set_params(steps=np.array(steps))

    def _raw_ticks(self, vmin, vmax):
        self._guess_steps(vmin, vmax)
        return MaxNLocator._raw_ticks(self, vmin, vmax)

    def bin_boundaries(self, vmin, vmax):
        self._guess_steps(vmin, vmax)
        return MaxNLocator.bin_boundaries(self, vmin, vmax)