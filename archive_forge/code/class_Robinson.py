from abc import ABCMeta
from collections import OrderedDict
from functools import lru_cache
import io
import json
import math
import warnings
import numpy as np
from pyproj import Transformer
from pyproj.exceptions import ProjError
import shapely.geometry as sgeom
from shapely.prepared import prep
import cartopy.trace
class Robinson(_WarpedRectangularProjection):
    """
    A Robinson projection.

    This projection is pseudocylindrical, and a compromise that is neither
    equal-area nor conformal. Parallels are unequally-spaced straight lines,
    and meridians are curved lines of no particular form.

    It is commonly used for "visually-appealing" world maps.

    """
    _handles_ellipses = False

    def __init__(self, central_longitude=0, globe=None, false_easting=None, false_northing=None):
        """
        Parameters
        ----------
        central_longitude: float, optional
            The central longitude. Defaults to 0.
        false_easting: float, optional
            X offset from planar origin in metres. Defaults to 0.
        false_northing: float, optional
            Y offset from planar origin in metres. Defaults to 0.
        globe: :class:`cartopy.crs.Globe`, optional
            If omitted, a default globe is created.

            .. note::
                This projection does not handle elliptical globes.

        """
        proj4_params = [('proj', 'robin'), ('lon_0', central_longitude)]
        super().__init__(proj4_params, central_longitude, false_easting=false_easting, false_northing=false_northing, globe=globe)
        self.threshold = 10000.0

    def transform_point(self, x, y, src_crs, trap=True):
        """
        Capture and handle any input NaNs, else invoke parent function,
        :meth:`_WarpedRectangularProjection.transform_point`.

        Needed because input NaNs can trigger a fatal error in the underlying
        implementation of the Robinson projection.

        Note
        ----
            Although the original can in fact translate (nan, lat) into
            (nan, y-value), this patched version doesn't support that.

        """
        if np.isnan(x) or np.isnan(y):
            result = (np.nan, np.nan)
        else:
            result = super().transform_point(x, y, src_crs, trap=trap)
        return result

    def transform_points(self, src_crs, x, y, z=None, trap=False):
        """
        Capture and handle NaNs in input points -- else as parent function,
        :meth:`_WarpedRectangularProjection.transform_points`.

        Needed because input NaNs can trigger a fatal error in the underlying
        implementation of the Robinson projection.

        Note
        ----
            Although the original can in fact translate (nan, lat) into
            (nan, y-value), this patched version doesn't support that.
            Instead, we invalidate any of the points that contain a NaN.

        """
        input_point_nans = np.isnan(x) | np.isnan(y)
        if z is not None:
            input_point_nans |= np.isnan(z)
        handle_nans = np.any(input_point_nans)
        if handle_nans:
            x[input_point_nans] = 0.0
            y[input_point_nans] = 0.0
            if z is not None:
                z[input_point_nans] = 0.0
        result = super().transform_points(src_crs, x, y, z, trap=trap)
        if handle_nans:
            result[input_point_nans] = np.nan
        return result