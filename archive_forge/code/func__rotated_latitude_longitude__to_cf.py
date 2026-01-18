import warnings
from pyproj._crs import Datum, Ellipsoid, PrimeMeridian
from pyproj.crs.coordinate_operation import (
from pyproj.crs.datum import CustomDatum, CustomEllipsoid, CustomPrimeMeridian
from pyproj.exceptions import CRSError
def _rotated_latitude_longitude__to_cf(conversion):
    """
    http://cfconventions.org/cf-conventions/cf-conventions.html#_rotated_pole
    """
    params = _to_dict(conversion)
    return {'grid_mapping_name': 'rotated_latitude_longitude', 'grid_north_pole_latitude': params['o_lat_p'], 'grid_north_pole_longitude': params['lon_0'] - 180, 'north_pole_grid_longitude': params['o_lon_p']}