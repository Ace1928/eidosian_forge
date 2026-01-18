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
class CRSLocal(threading.local):
    """
    Threading local instance for cython CRS class.

    For more details, see:
    https://github.com/pyproj4/pyproj/issues/782
    """

    def __init__(self):
        self.crs = None
        super().__init__()