import warnings
from typing import Any
from pyproj._crs import CoordinateOperation
from pyproj.exceptions import CRSError
class RotatedLatitudeLongitudeConversion(CoordinateOperation):
    """
    .. versionadded:: 2.5.0

    Class for constructing the Rotated Latitude Longitude conversion.

    :ref:`PROJ docs <ob_tran>`
    """

    def __new__(cls, o_lat_p: float, o_lon_p: float, lon_0: float=0.0):
        """
        Parameters
        ----------
        o_lat_p: float
            Latitude of the North pole of the unrotated source CRS,
            expressed in the rotated geographic CRS.
        o_lon_p: float
            Longitude of the North pole of the unrotated source CRS,
            expressed in the rotated geographic CRS.
        lon_0: float, default=0.0
            Longitude of projection center.

        """
        rot_latlon_json = {'$schema': 'https://proj.org/schemas/v0.2/projjson.schema.json', 'type': 'Conversion', 'name': 'unknown', 'method': {'name': 'PROJ ob_tran o_proj=longlat'}, 'parameters': [{'name': 'o_lat_p', 'value': o_lat_p, 'unit': 'degree'}, {'name': 'o_lon_p', 'value': o_lon_p, 'unit': 'degree'}, {'name': 'lon_0', 'value': lon_0, 'unit': 'degree'}]}
        return cls.from_json_dict(rot_latlon_json)