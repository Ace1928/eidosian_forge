import warnings
from pyproj._crs import Datum, Ellipsoid, PrimeMeridian
from pyproj.crs.coordinate_operation import (
from pyproj.crs.datum import CustomDatum, CustomEllipsoid, CustomPrimeMeridian
from pyproj.exceptions import CRSError
def _lambert_cylindrical_equal_area__to_cf(conversion):
    """
    http://cfconventions.org/cf-conventions/cf-conventions.html#_lambert_cylindrical_equal_area
    """
    params = _to_dict(conversion)
    return {'grid_mapping_name': 'lambert_cylindrical_equal_area', 'standard_parallel': params['latitude_of_1st_standard_parallel'], 'longitude_of_central_meridian': params['longitude_of_natural_origin'], 'false_easting': params['false_easting'], 'false_northing': params['false_northing']}