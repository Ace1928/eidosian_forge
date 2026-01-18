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
class GeocentricCRS(CustomConstructorCRS):
    """
    .. versionadded:: 3.2.0

    This class is for building a Geocentric CRS
    """
    _expected_types = ('Geocentric CRS',)

    def __init__(self, name: str='undefined', datum: Any='urn:ogc:def:datum:EPSG::6326') -> None:
        """
        Parameters
        ----------
        name: str, default="undefined"
            Name of the CRS.
        datum: Any, default="urn:ogc:def:datum:EPSG::6326"
            Anything accepted by :meth:`pyproj.crs.Datum.from_user_input` or
            a :class:`pyproj.crs.datum.CustomDatum`.
        """
        geocentric_crs_json = {'$schema': 'https://proj.org/schemas/v0.2/projjson.schema.json', 'type': 'GeodeticCRS', 'name': name, 'datum': Datum.from_user_input(datum).to_json_dict(), 'coordinate_system': {'subtype': 'Cartesian', 'axis': [{'name': 'Geocentric X', 'abbreviation': 'X', 'direction': 'geocentricX', 'unit': 'metre'}, {'name': 'Geocentric Y', 'abbreviation': 'Y', 'direction': 'geocentricY', 'unit': 'metre'}, {'name': 'Geocentric Z', 'abbreviation': 'Z', 'direction': 'geocentricZ', 'unit': 'metre'}]}}
        super().__init__(geocentric_crs_json)