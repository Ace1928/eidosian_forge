import warnings
from pyproj._crs import Datum, Ellipsoid, PrimeMeridian
from pyproj.crs.coordinate_operation import (
from pyproj.crs.datum import CustomDatum, CustomEllipsoid, CustomPrimeMeridian
from pyproj.exceptions import CRSError
def _get_standard_parallels(standard_parallel):
    standard_parallel = _try_list_if_string(standard_parallel)
    try:
        first_parallel = float(standard_parallel)
        second_parallel = None
    except TypeError:
        first_parallel, second_parallel = standard_parallel
    return (first_parallel, second_parallel)