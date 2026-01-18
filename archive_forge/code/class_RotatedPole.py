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
class RotatedPole(_CylindricalProjection):
    """
    A rotated latitude/longitude projected coordinate system
    with cylindrical topology and projected distance.

    Coordinates are measured in projection metres.

    The class uses proj to perform an ob_tran operation, using the
    pole_longitude to set a lon_0 then performing two rotations based on
    pole_latitude and central_rotated_longitude.
    This is equivalent to setting the new pole to a location defined by
    the pole_latitude and pole_longitude values in the GeogCRS defined by
    globe, then rotating this new CRS about it's pole using the
    central_rotated_longitude value.

    """

    def __init__(self, pole_longitude=0.0, pole_latitude=90.0, central_rotated_longitude=0.0, globe=None):
        """
        Parameters
        ----------
        pole_longitude: optional
            Pole longitude position, in unrotated degrees. Defaults to 0.
        pole_latitude: optional
            Pole latitude position, in unrotated degrees. Defaults to 0.
        central_rotated_longitude: optional
            Longitude rotation about the new pole, in degrees. Defaults to 0.
        globe: optional
            An optional :class:`cartopy.crs.Globe`. Defaults to a "WGS84"
            datum.

        """
        globe = globe or Globe(semimajor_axis=WGS84_SEMIMAJOR_AXIS)
        proj4_params = [('proj', 'ob_tran'), ('o_proj', 'latlon'), ('o_lon_p', central_rotated_longitude), ('o_lat_p', pole_latitude), ('lon_0', 180 + pole_longitude), ('to_meter', math.radians(1) * (globe.semimajor_axis or WGS84_SEMIMAJOR_AXIS))]
        super().__init__(proj4_params, 180, 90, globe=globe)