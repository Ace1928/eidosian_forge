from abc import ABCMeta, abstractmethod
import numpy as np
import shapely.geometry as sgeom
import cartopy.crs
import cartopy.io.shapereader as shapereader
class AdaptiveScaler(Scaler):
    """
    Automatically select scale of geometries based on extent of plotted axes.
    """

    def __init__(self, default_scale, limits):
        """
        Parameters
        ----------
        default_scale
            Coarsest scale used as default when plot is at maximum extent.

        limits
            Scale-extent pairs at which scale of geometries change. Must be a
            tuple of tuples ordered from coarsest to finest scales. Limit
            values are the upper bounds for their corresponding scale.

        Example
        -------

        >>> s = AdaptiveScaler('coarse',
        ...           (('intermediate', 30), ('fine', 10)))
        >>> s.scale_from_extent([-180, 180, -90, 90])
        'coarse'
        >>> s.scale_from_extent([-5, 6, 45, 56])
        'intermediate'
        >>> s.scale_from_extent([-5, 5, 45, 56])
        'fine'

        """
        super().__init__(default_scale)
        self._default_scale = default_scale
        self._limits = limits

    def scale_from_extent(self, extent):
        scale = self._default_scale
        if extent is not None:
            width = abs(extent[1] - extent[0])
            height = abs(extent[3] - extent[2])
            min_extent = min(width, height)
            if min_extent != 0:
                for scale_candidate, upper_bound in self._limits:
                    if min_extent <= upper_bound:
                        scale = scale_candidate
                    else:
                        break
        self._scale = scale
        return self._scale