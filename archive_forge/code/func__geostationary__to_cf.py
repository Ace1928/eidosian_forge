import warnings
from pyproj._crs import Datum, Ellipsoid, PrimeMeridian
from pyproj.crs.coordinate_operation import (
from pyproj.crs.datum import CustomDatum, CustomEllipsoid, CustomPrimeMeridian
from pyproj.exceptions import CRSError
def _geostationary__to_cf(conversion):
    """
    http://cfconventions.org/cf-conventions/cf-conventions.html#_geostationary_projection
    """
    params = _to_dict(conversion)
    sweep_angle_axis = 'y'
    if conversion.method_name.lower().replace(' ', '_').endswith('(sweep_x)'):
        sweep_angle_axis = 'x'
    return {'grid_mapping_name': 'geostationary', 'sweep_angle_axis': sweep_angle_axis, 'perspective_point_height': params['satellite_height'], 'latitude_of_projection_origin': params.get('latitude_of_natural_origin', 0.0), 'longitude_of_projection_origin': params['longitude_of_natural_origin'], 'false_easting': params['false_easting'], 'false_northing': params['false_northing']}