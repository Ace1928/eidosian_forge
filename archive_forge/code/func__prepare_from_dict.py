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
def _prepare_from_dict(projparams: dict, allow_json: bool=True) -> str:
    if not isinstance(projparams, dict):
        raise CRSError('CRS input is not a dict')
    if 'proj' not in projparams and 'init' not in projparams and allow_json:
        return json.dumps(projparams)
    pjargs = []
    for key, value in projparams.items():
        if isinstance(value, (list, tuple)):
            value = ','.join([str(val) for val in value])
        if value is None or str(value) == 'True':
            pjargs.append(f'+{key}')
        elif str(value) == 'False':
            pass
        else:
            pjargs.append(f'+{key}={value}')
    return _prepare_from_string(' '.join(pjargs))