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
class BoundCRS(CustomConstructorCRS):
    """
    .. versionadded:: 2.5.0

    This class is for building a Bound CRS.
    """
    _expected_types = ('Bound CRS',)

    def __init__(self, source_crs: Any, target_crs: Any, transformation: Any) -> None:
        """
        Parameters
        ----------
        source_crs: Any
            Input to create a source CRS.
        target_crs: Any
            Input to create the target CRS.
        transformation: Any
            Input to create the transformation.
        """
        bound_crs_json = {'$schema': 'https://proj.org/schemas/v0.2/projjson.schema.json', 'type': 'BoundCRS', 'source_crs': CRS.from_user_input(source_crs).to_json_dict(), 'target_crs': CRS.from_user_input(target_crs).to_json_dict(), 'transformation': CoordinateOperation.from_user_input(transformation).to_json_dict()}
        super().__init__(bound_crs_json)