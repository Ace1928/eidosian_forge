import warnings
from pyproj._crs import Datum, Ellipsoid, PrimeMeridian
from pyproj.crs.coordinate_operation import (
from pyproj.crs.datum import CustomDatum, CustomEllipsoid, CustomPrimeMeridian
from pyproj.exceptions import CRSError
def _oblique_mercator__to_cf(conversion):
    """
    http://cfconventions.org/cf-conventions/cf-conventions.html#_oblique_mercator
    """
    params = _to_dict(conversion)
    if params['angle_from_rectified_to_skew_grid'] != 0:
        warnings.warn('angle from rectified to skew grid parameter lost in conversion to CF')
    return {'grid_mapping_name': 'oblique_mercator', 'latitude_of_projection_origin': params['latitude_of_projection_centre'], 'longitude_of_projection_origin': params['longitude_of_projection_centre'], 'azimuth_of_central_line': params['azimuth_of_initial_line'], 'scale_factor_at_projection_origin': params['scale_factor_on_initial_line'], 'false_easting': params['easting_at_projection_centre'], 'false_northing': params['northing_at_projection_centre']}