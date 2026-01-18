from abc import ABCMeta, abstractmethod
import numpy as np
import shapely.geometry as sgeom
import cartopy.crs
import cartopy.io.shapereader as shapereader
class Scaler:
    """
    General object for handling the scale of the geometries used in a Feature.
    """

    def __init__(self, scale):
        self._scale = scale

    @property
    def scale(self):
        return self._scale

    def scale_from_extent(self, extent):
        """
        Given an extent, update the scale.

        Parameters
        ----------
        extent
            The boundaries of the plotted area of a projection. The
            coordinate system of the extent should be constant, and at the
            same scale as the scales argument in the constructor.

        """
        return self._scale