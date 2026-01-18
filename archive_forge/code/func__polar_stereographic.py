import warnings
from pyproj._crs import Datum, Ellipsoid, PrimeMeridian
from pyproj.crs.coordinate_operation import (
from pyproj.crs.datum import CustomDatum, CustomEllipsoid, CustomPrimeMeridian
from pyproj.exceptions import CRSError
def _polar_stereographic(cf_params):
    """
    http://cfconventions.org/cf-conventions/cf-conventions.html#polar-stereographic
    """
    if 'standard_parallel' in cf_params:
        return PolarStereographicBConversion(latitude_standard_parallel=cf_params['standard_parallel'], longitude_origin=cf_params['straight_vertical_longitude_from_pole'], false_easting=cf_params.get('false_easting', 0.0), false_northing=cf_params.get('false_northing', 0.0))
    return PolarStereographicAConversion(latitude_natural_origin=cf_params['latitude_of_projection_origin'], longitude_natural_origin=cf_params['straight_vertical_longitude_from_pole'], false_easting=cf_params.get('false_easting', 0.0), false_northing=cf_params.get('false_northing', 0.0), scale_factor_natural_origin=cf_params.get('scale_factor_at_projection_origin', 1.0))