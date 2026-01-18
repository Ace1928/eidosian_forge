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
class DerivedGeographicCRS(CustomConstructorCRS):
    """
    .. versionadded:: 2.5.0

    This class is for building a Derived Geographic CRS
    """
    _expected_types = ('Derived Geographic CRS', 'Derived Geographic 2D CRS', 'Derived Geographic 3D CRS')

    def __init__(self, base_crs: Any, conversion: Any, ellipsoidal_cs: Optional[Any]=None, name: str='undefined') -> None:
        """
        Parameters
        ----------
        base_crs: Any
            Input to create the Geodetic CRS, a :class:`GeographicCRS` or
            anything accepted by :meth:`pyproj.crs.CRS.from_user_input`.
        conversion: Any
            Anything accepted by :meth:`pyproj.crs.CoordinateSystem.from_user_input`
            or a conversion from :ref:`coordinate_operation`.
        ellipsoidal_cs: Any, optional
            Input to create an Ellipsoidal Coordinate System.
            Anything accepted by :meth:`pyproj.crs.CoordinateSystem.from_user_input`
            or an Ellipsoidal Coordinate System created from :ref:`coordinate_system`.
        name: str, default="undefined"
            Name of the CRS.
        """
        derived_geographic_crs_json = {'$schema': 'https://proj.org/schemas/v0.2/projjson.schema.json', 'type': 'DerivedGeographicCRS', 'name': name, 'base_crs': CRS.from_user_input(base_crs).to_json_dict(), 'conversion': CoordinateOperation.from_user_input(conversion).to_json_dict(), 'coordinate_system': CoordinateSystem.from_user_input(ellipsoidal_cs or Ellipsoidal2DCS()).to_json_dict()}
        super().__init__(derived_geographic_crs_json)