import matplotlib as mpl
from matplotlib.ticker import Formatter, MaxNLocator
import numpy as np
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxes
class LatitudeLocator(LongitudeLocator):
    """
    A locator for latitudes that works even at very small scale.

    Parameters
    ----------
    dms: bool
        Allow the locator to stop on minutes and seconds (False by default)
    """

    def tick_values(self, vmin, vmax):
        vmin = max(vmin, -90.0)
        vmax = min(vmax, 90.0)
        return LongitudeLocator.tick_values(self, vmin, vmax)

    def _guess_steps(self, vmin, vmax):
        vmin = max(vmin, -90.0)
        vmax = min(vmax, 90.0)
        LongitudeLocator._guess_steps(self, vmin, vmax)

    def _raw_ticks(self, vmin, vmax):
        ticks = LongitudeLocator._raw_ticks(self, vmin, vmax)
        return [t for t in ticks if -90 <= t <= 90]

    def bin_boundaries(self, vmin, vmax):
        ticks = LongitudeLocator.bin_boundaries(self, vmin, vmax)
        return [t for t in ticks if -90 <= t <= 90]