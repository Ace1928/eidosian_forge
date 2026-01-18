import warnings
from pyproj._crs import Datum, Ellipsoid, PrimeMeridian
from pyproj.crs.coordinate_operation import (
from pyproj.crs.datum import CustomDatum, CustomEllipsoid, CustomPrimeMeridian
from pyproj.exceptions import CRSError
def _orthographic__to_cf(conversion):
    """
    http://cfconventions.org/cf-conventions/cf-conventions.html#_orthographic
    """
    params = _to_dict(conversion)
    return {'grid_mapping_name': 'orthographic', 'latitude_of_projection_origin': params['latitude_of_natural_origin'], 'longitude_of_projection_origin': params['longitude_of_natural_origin'], 'false_easting': params['false_easting'], 'false_northing': params['false_northing']}