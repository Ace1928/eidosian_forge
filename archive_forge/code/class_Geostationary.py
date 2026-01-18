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
class Geostationary(_Satellite):
    """
    A view appropriate for satellites in Geostationary Earth orbit.

    Perspective view looking directly down from above a point on the equator.

    In this projection, the projected coordinates are scanning angles measured
    from the satellite looking directly downward, multiplied by the height of
    the satellite.

    """

    def __init__(self, central_longitude=0.0, satellite_height=35785831, false_easting=0, false_northing=0, globe=None, sweep_axis='y'):
        """
        Parameters
        ----------
        central_longitude: float, optional
            The central longitude. Defaults to 0.
        satellite_height: float, optional
            The height of the satellite. Defaults to 35785831 metres
            (true geostationary orbit).
        false_easting:
            X offset from planar origin in metres. Defaults to 0.
        false_northing:
            Y offset from planar origin in metres. Defaults to 0.
        globe: :class:`cartopy.crs.Globe`, optional
            If omitted, a default globe is created.
        sweep_axis: 'x' or 'y', optional. Defaults to 'y'.
            Controls which axis is scanned first, and thus which angle is
            applied first. The default is appropriate for Meteosat, while
            'x' should be used for GOES.
        """
        super().__init__(projection='geos', satellite_height=satellite_height, central_longitude=central_longitude, central_latitude=0.0, false_easting=false_easting, false_northing=false_northing, globe=globe, sweep_axis=sweep_axis)
        a = float(self.ellipsoid.semi_major_metre or WGS84_SEMIMAJOR_AXIS)
        b = float(self.ellipsoid.semi_minor_metre or WGS84_SEMIMINOR_AXIS)
        h = float(satellite_height)
        angleA = np.linspace(0, -2 * np.pi, 91)
        th = np.arctan(a / b * np.tan(angleA))
        r = np.hypot(a * np.cos(th), b * np.sin(th))
        sat_dist = a + h
        sin_c = r / np.sqrt(sat_dist ** 2 - a ** 2 + r ** 2)
        tan_c = r / np.sqrt(sat_dist ** 2 - a ** 2)
        coords = np.vstack([np.arctan(np.cos(angleA) * tan_c), np.arcsin(np.sin(angleA) * sin_c)])
        coords *= h
        coords += np.array([[false_easting], [false_northing]])
        self._set_boundary(coords)