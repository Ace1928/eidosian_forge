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
class GeographicCRS(CustomConstructorCRS):
    """
    .. versionadded:: 2.5.0

    This class is for building a Geographic CRS
    """
    _expected_types = ('Geographic CRS', 'Geographic 2D CRS', 'Geographic 3D CRS')

    def __init__(self, name: str='undefined', datum: Any='urn:ogc:def:ensemble:EPSG::6326', ellipsoidal_cs: Optional[Any]=None) -> None:
        """
        Parameters
        ----------
        name: str, default="undefined"
            Name of the CRS.
        datum: Any, default="urn:ogc:def:ensemble:EPSG::6326"
            Anything accepted by :meth:`pyproj.crs.Datum.from_user_input` or
            a :class:`pyproj.crs.datum.CustomDatum`.
        ellipsoidal_cs: Any, optional
            Input to create an Ellipsoidal Coordinate System.
            Anything accepted by :meth:`pyproj.crs.CoordinateSystem.from_user_input`
            or an Ellipsoidal Coordinate System created from :ref:`coordinate_system`.
        """
        datum = Datum.from_user_input(datum).to_json_dict()
        geographic_crs_json = {'$schema': 'https://proj.org/schemas/v0.2/projjson.schema.json', 'type': 'GeographicCRS', 'name': name, 'coordinate_system': CoordinateSystem.from_user_input(ellipsoidal_cs or Ellipsoidal2DCS()).to_json_dict()}
        if datum['type'] == 'DatumEnsemble':
            geographic_crs_json['datum_ensemble'] = datum
        else:
            geographic_crs_json['datum'] = datum
        super().__init__(geographic_crs_json)