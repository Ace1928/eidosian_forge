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
class VerticalCRS(CustomConstructorCRS):
    """
    .. versionadded:: 2.5.0

    This class is for building a Vetical CRS.

    .. warning:: geoid_model support only exists in PROJ >= 6.3.0

    """
    _expected_types = ('Vertical CRS',)

    def __init__(self, name: str, datum: Any, vertical_cs: Optional[Any]=None, geoid_model: Optional[str]=None) -> None:
        """
        Parameters
        ----------
        name: str
            The name of the Vertical CRS (e.g. NAVD88 height).
        datum: Any
            Anything accepted by :meth:`pyproj.crs.Datum.from_user_input`
        vertical_cs: Any, optional
            Input to create a Vertical Coordinate System accepted by
            :meth:`pyproj.crs.CoordinateSystem.from_user_input`
            or :class:`pyproj.crs.coordinate_system.VerticalCS`
        geoid_model: str, optional
            The name of the GEOID Model (e.g. GEOID12B).
        """
        vert_crs_json = {'$schema': 'https://proj.org/schemas/v0.2/projjson.schema.json', 'type': 'VerticalCRS', 'name': name, 'datum': Datum.from_user_input(datum).to_json_dict(), 'coordinate_system': CoordinateSystem.from_user_input(vertical_cs or VerticalCS()).to_json_dict()}
        if geoid_model is not None:
            vert_crs_json['geoid_model'] = {'name': geoid_model}
        super().__init__(vert_crs_json)