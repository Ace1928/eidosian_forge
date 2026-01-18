import warnings
from pyproj._crs import Datum, Ellipsoid, PrimeMeridian
from pyproj.crs.coordinate_operation import (
from pyproj.crs.datum import CustomDatum, CustomEllipsoid, CustomPrimeMeridian
from pyproj.exceptions import CRSError
def _albers_conical_equal_area(cf_params):
    """
    http://cfconventions.org/cf-conventions/cf-conventions.html#_albers_equal_area
    """
    first_parallel, second_parallel = _get_standard_parallels(cf_params['standard_parallel'])
    return AlbersEqualAreaConversion(latitude_first_parallel=first_parallel, latitude_second_parallel=second_parallel or 0.0, latitude_false_origin=cf_params.get('latitude_of_projection_origin', 0.0), longitude_false_origin=cf_params.get('longitude_of_central_meridian', 0.0), easting_false_origin=cf_params.get('false_easting', 0.0), northing_false_origin=cf_params.get('false_northing', 0.0))