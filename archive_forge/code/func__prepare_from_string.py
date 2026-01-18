import json
import re
import threading
import warnings
from typing import Any, Callable, Optional, Union
from pyproj._crs import (
from pyproj.crs._cf1x8 import (
from pyproj.crs.coordinate_operation import ToWGS84Transformation
from pyproj.crs.coordinate_system import Cartesian2DCS, Ellipsoidal2DCS, VerticalCS
from pyproj.enums import ProjVersion, WktVersion
from pyproj.exceptions import CRSError
from pyproj.geod import Geod
def _prepare_from_string(in_crs_string: str) -> str:
    if not isinstance(in_crs_string, str):
        raise CRSError('CRS input is not a string')
    if not in_crs_string:
        raise CRSError(f'CRS string is empty or invalid: {in_crs_string!r}')
    if '{' in in_crs_string:
        try:
            crs_dict = json.loads(in_crs_string, strict=False)
        except ValueError as err:
            raise CRSError('CRS appears to be JSON but is not valid') from err
        if not crs_dict:
            raise CRSError('CRS is empty JSON')
        in_crs_string = _prepare_from_dict(crs_dict)
    elif is_proj(in_crs_string):
        in_crs_string = _prepare_from_proj_string(in_crs_string)
    return in_crs_string