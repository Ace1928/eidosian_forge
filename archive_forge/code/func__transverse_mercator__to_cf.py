import warnings
from pyproj._crs import Datum, Ellipsoid, PrimeMeridian
from pyproj.crs.coordinate_operation import (
from pyproj.crs.datum import CustomDatum, CustomEllipsoid, CustomPrimeMeridian
from pyproj.exceptions import CRSError
def _transverse_mercator__to_cf(conversion):
    """
    http://cfconventions.org/cf-conventions/cf-conventions.html#_transverse_mercator
    """
    params = _to_dict(conversion)
    return {'grid_mapping_name': 'transverse_mercator', 'latitude_of_projection_origin': params['latitude_of_natural_origin'], 'longitude_of_central_meridian': params['longitude_of_natural_origin'], 'false_easting': params['false_easting'], 'false_northing': params['false_northing'], 'scale_factor_at_central_meridian': params['scale_factor_at_natural_origin']}