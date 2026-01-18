import warnings
from pyproj._crs import Datum, Ellipsoid, PrimeMeridian
from pyproj.crs.coordinate_operation import (
from pyproj.crs.datum import CustomDatum, CustomEllipsoid, CustomPrimeMeridian
from pyproj.exceptions import CRSError
def _vertical_perspective(cf_params):
    """
    http://cfconventions.org/cf-conventions/cf-conventions.html#vertical-perspective
    """
    return VerticalPerspectiveConversion(viewpoint_height=cf_params['perspective_point_height'], latitude_topocentric_origin=cf_params.get('latitude_of_projection_origin', 0.0), longitude_topocentric_origin=cf_params.get('longitude_of_projection_origin', 0.0), false_easting=cf_params.get('false_easting', 0.0), false_northing=cf_params.get('false_northing', 0.0))