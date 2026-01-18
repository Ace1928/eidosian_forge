import numpy as np
import math
from mpl_toolkits.axisartist.grid_finder import ExtremeFinderSimple
class ExtremeFinderCycle(ExtremeFinderSimple):

    def __init__(self, nx, ny, lon_cycle=360.0, lat_cycle=None, lon_minmax=None, lat_minmax=(-90, 90)):
        """
        This subclass handles the case where one or both coordinates should be
        taken modulo 360, or be restricted to not exceed a specific range.

        Parameters
        ----------
        nx, ny : int
            The number of samples in each direction.

        lon_cycle, lat_cycle : 360 or None
            If not None, values in the corresponding direction are taken modulo
            *lon_cycle* or *lat_cycle*; in theory this can be any number but
            the implementation actually assumes that it is 360 (if not None);
            other values give nonsensical results.

            This is done by "unwrapping" the transformed grid coordinates so
            that jumps are less than a half-cycle; then normalizing the span to
            no more than a full cycle.

            For example, if values are in the union of the [0, 2] and
            [358, 360] intervals (typically, angles measured modulo 360), the
            values in the second interval are normalized to [-2, 0] instead so
            that the values now cover [-2, 2].  If values are in a range of
            [5, 1000], this gets normalized to [5, 365].

        lon_minmax, lat_minmax : (float, float) or None
            If not None, the computed bounding box is clipped to the given
            range in the corresponding direction.
        """
        self.nx, self.ny = (nx, ny)
        self.lon_cycle, self.lat_cycle = (lon_cycle, lat_cycle)
        self.lon_minmax = lon_minmax
        self.lat_minmax = lat_minmax

    def __call__(self, transform_xy, x1, y1, x2, y2):
        x, y = np.meshgrid(np.linspace(x1, x2, self.nx), np.linspace(y1, y2, self.ny))
        lon, lat = transform_xy(np.ravel(x), np.ravel(y))
        with np.errstate(invalid='ignore'):
            if self.lon_cycle is not None:
                lon0 = np.nanmin(lon)
                lon -= 360.0 * (lon - lon0 > 180.0)
            if self.lat_cycle is not None:
                lat0 = np.nanmin(lat)
                lat -= 360.0 * (lat - lat0 > 180.0)
        lon_min, lon_max = (np.nanmin(lon), np.nanmax(lon))
        lat_min, lat_max = (np.nanmin(lat), np.nanmax(lat))
        lon_min, lon_max, lat_min, lat_max = self._add_pad(lon_min, lon_max, lat_min, lat_max)
        if self.lon_cycle:
            lon_max = min(lon_max, lon_min + self.lon_cycle)
        if self.lat_cycle:
            lat_max = min(lat_max, lat_min + self.lat_cycle)
        if self.lon_minmax is not None:
            min0 = self.lon_minmax[0]
            lon_min = max(min0, lon_min)
            max0 = self.lon_minmax[1]
            lon_max = min(max0, lon_max)
        if self.lat_minmax is not None:
            min0 = self.lat_minmax[0]
            lat_min = max(min0, lat_min)
            max0 = self.lat_minmax[1]
            lat_max = min(max0, lat_max)
        return (lon_min, lon_max, lat_min, lat_max)