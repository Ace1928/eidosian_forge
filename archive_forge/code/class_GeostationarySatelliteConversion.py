import warnings
from typing import Any
from pyproj._crs import CoordinateOperation
from pyproj.exceptions import CRSError
class GeostationarySatelliteConversion(CoordinateOperation):
    """
    .. versionadded:: 2.5.0

    Class for constructing the Geostationary Satellite conversion.

    :ref:`PROJ docs <geos>`
    """

    def __new__(cls, sweep_angle_axis: str, satellite_height: float, latitude_natural_origin: float=0.0, longitude_natural_origin: float=0.0, false_easting: float=0.0, false_northing: float=0.0):
        """
        Parameters
        ----------
        sweep_angle_axis: str
            Sweep angle axis of the viewing instrument. Valid options are “X” and “Y”.
        satellite_height: float
            Satellite height.
        latitude_natural_origin: float, default=0.0
            Latitude of projection center (lat_0).
        longitude_natural_origin: float, default=0.0
            Longitude of projection center (lon_0).
        false_easting: float, default=0.0
            False easting (x_0).
        false_northing: float, default=0.0
            False northing (y_0).

        """
        sweep_angle_axis = sweep_angle_axis.strip().upper()
        valid_sweep_axis = ('X', 'Y')
        if sweep_angle_axis not in valid_sweep_axis:
            raise CRSError(f'sweep_angle_axis only supports {valid_sweep_axis}')
        if latitude_natural_origin != 0:
            warnings.warn('The latitude of natural origin (lat_0) is not used within PROJ. It is only supported for exporting to the WKT or PROJ JSON formats.')
        geos_json = {'$schema': 'https://proj.org/schemas/v0.2/projjson.schema.json', 'type': 'Conversion', 'name': 'unknown', 'method': {'name': f'Geostationary Satellite (Sweep {sweep_angle_axis})'}, 'parameters': [{'name': 'Satellite height', 'value': satellite_height, 'unit': 'metre'}, {'name': 'Latitude of natural origin', 'value': latitude_natural_origin, 'unit': 'degree', 'id': {'authority': 'EPSG', 'code': 8801}}, {'name': 'Longitude of natural origin', 'value': longitude_natural_origin, 'unit': 'degree', 'id': {'authority': 'EPSG', 'code': 8802}}, {'name': 'False easting', 'value': false_easting, 'unit': 'metre', 'id': {'authority': 'EPSG', 'code': 8806}}, {'name': 'False northing', 'value': false_northing, 'unit': 'metre', 'id': {'authority': 'EPSG', 'code': 8807}}]}
        return cls.from_json_dict(geos_json)