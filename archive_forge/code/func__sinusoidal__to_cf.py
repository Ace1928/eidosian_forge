import warnings
from pyproj._crs import Datum, Ellipsoid, PrimeMeridian
from pyproj.crs.coordinate_operation import (
from pyproj.crs.datum import CustomDatum, CustomEllipsoid, CustomPrimeMeridian
from pyproj.exceptions import CRSError
def _sinusoidal__to_cf(conversion):
    """
    http://cfconventions.org/cf-conventions/cf-conventions.html#_sinusoidal
    """
    params = _to_dict(conversion)
    return {'grid_mapping_name': 'sinusoidal', 'longitude_of_projection_origin': params['longitude_of_natural_origin'], 'false_easting': params['false_easting'], 'false_northing': params['false_northing']}