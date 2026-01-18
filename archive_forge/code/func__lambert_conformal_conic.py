import warnings
from pyproj._crs import Datum, Ellipsoid, PrimeMeridian
from pyproj.crs.coordinate_operation import (
from pyproj.crs.datum import CustomDatum, CustomEllipsoid, CustomPrimeMeridian
from pyproj.exceptions import CRSError
def _lambert_conformal_conic(cf_params):
    """
    http://cfconventions.org/cf-conventions/cf-conventions.html#_lambert_conformal
    """
    first_parallel, second_parallel = _get_standard_parallels(cf_params['standard_parallel'])
    if second_parallel is not None:
        return LambertConformalConic2SPConversion(latitude_first_parallel=first_parallel, latitude_second_parallel=second_parallel, latitude_false_origin=cf_params.get('latitude_of_projection_origin', 0.0), longitude_false_origin=cf_params.get('longitude_of_central_meridian', 0.0), easting_false_origin=cf_params.get('false_easting', 0.0), northing_false_origin=cf_params.get('false_northing', 0.0))
    return LambertConformalConic1SPConversion(latitude_natural_origin=first_parallel, longitude_natural_origin=cf_params.get('longitude_of_central_meridian', 0.0), false_easting=cf_params.get('false_easting', 0.0), false_northing=cf_params.get('false_northing', 0.0))