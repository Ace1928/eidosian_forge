import warnings
from pyproj._crs import Datum, Ellipsoid, PrimeMeridian
from pyproj.crs.coordinate_operation import (
from pyproj.crs.datum import CustomDatum, CustomEllipsoid, CustomPrimeMeridian
from pyproj.exceptions import CRSError
def _mercator__to_cf(conversion):
    """
    http://cfconventions.org/cf-conventions/cf-conventions.html#_mercator
    """
    params = _to_dict(conversion)
    if conversion.method_name.lower().replace(' ', '_').endswith('(variant_a)'):
        return {'grid_mapping_name': 'mercator', 'standard_parallel': params['latitude_of_natural_origin'], 'longitude_of_projection_origin': params['longitude_of_natural_origin'], 'false_easting': params['false_easting'], 'false_northing': params['false_northing'], 'scale_factor_at_projection_origin': params['scale_factor_at_natural_origin']}
    return {'grid_mapping_name': 'mercator', 'standard_parallel': params['latitude_of_1st_standard_parallel'], 'longitude_of_projection_origin': params['longitude_of_natural_origin'], 'false_easting': params['false_easting'], 'false_northing': params['false_northing']}