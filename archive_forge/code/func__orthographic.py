import warnings
from pyproj._crs import Datum, Ellipsoid, PrimeMeridian
from pyproj.crs.coordinate_operation import (
from pyproj.crs.datum import CustomDatum, CustomEllipsoid, CustomPrimeMeridian
from pyproj.exceptions import CRSError
def _orthographic(cf_params):
    """
    http://cfconventions.org/cf-conventions/cf-conventions.html#_orthographic
    """
    return OrthographicConversion(latitude_natural_origin=cf_params.get('latitude_of_projection_origin', 0.0), longitude_natural_origin=cf_params.get('longitude_of_projection_origin', 0.0), false_easting=cf_params.get('false_easting', 0.0), false_northing=cf_params.get('false_northing', 0.0))