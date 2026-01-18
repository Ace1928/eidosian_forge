import warnings
from typing import Any
from pyproj._crs import CoordinateOperation
from pyproj.exceptions import CRSError
class PolarStereographicBConversion(CoordinateOperation):
    """
    .. versionadded:: 2.5.0

    Class for constructing the Polar Stereographic B conversion.

    :ref:`PROJ docs <stere>`
    """

    def __new__(cls, latitude_standard_parallel: float=0.0, longitude_origin: float=0.0, false_easting: float=0.0, false_northing: float=0.0):
        """
        Parameters
        ----------
        latitude_standard_parallel: float, default=0.0
            Latitude of standard parallel (lat_ts).
        longitude_origin: float, default=0.0
            Longitude of origin (lon_0).
        false_easting: float, default=0.0
            False easting (x_0).
        false_northing: float, default=0.0
            False northing (y_0).

        """
        stere_json = {'$schema': 'https://proj.org/schemas/v0.2/projjson.schema.json', 'type': 'Conversion', 'name': 'unknown', 'method': {'name': 'Polar Stereographic (variant B)', 'id': {'authority': 'EPSG', 'code': 9829}}, 'parameters': [{'name': 'Latitude of standard parallel', 'value': latitude_standard_parallel, 'unit': 'degree', 'id': {'authority': 'EPSG', 'code': 8832}}, {'name': 'Longitude of origin', 'value': longitude_origin, 'unit': 'degree', 'id': {'authority': 'EPSG', 'code': 8833}}, {'name': 'False easting', 'value': false_easting, 'unit': 'metre', 'id': {'authority': 'EPSG', 'code': 8806}}, {'name': 'False northing', 'value': false_northing, 'unit': 'metre', 'id': {'authority': 'EPSG', 'code': 8807}}]}
        return cls.from_json_dict(stere_json)