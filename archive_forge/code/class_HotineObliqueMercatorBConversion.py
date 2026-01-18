import warnings
from typing import Any
from pyproj._crs import CoordinateOperation
from pyproj.exceptions import CRSError
class HotineObliqueMercatorBConversion(CoordinateOperation):
    """
    .. versionadded:: 2.5.0

    Class for constructing the Hotine Oblique Mercator (variant B) conversion.

    :ref:`PROJ docs <omerc>`
    """

    def __new__(cls, latitude_projection_centre: float, longitude_projection_centre: float, azimuth_initial_line: float, angle_from_rectified_to_skew_grid: float, scale_factor_on_initial_line: float=1.0, easting_projection_centre: float=0.0, northing_projection_centre: float=0.0):
        """
        Parameters
        ----------
        latitude_projection_centre: float
            Latitude of projection centre (lat_0).
        longitude_projection_centre: float
            Longitude of projection centre (lonc).
        azimuth_initial_line: float
            Azimuth of initial line (alpha).
        angle_from_rectified_to_skew_grid: float
            Angle from Rectified to Skew Grid (gamma).
        scale_factor_on_initial_line: float, default=1.0
            Scale factor on initial line (k or k_0).
        easting_projection_centre: float, default=0.0
            Easting at projection centre (x_0).
        northing_projection_centre: float, default=0.0
            Northing at projection centre (y_0).
        """
        omerc_json = {'$schema': 'https://proj.org/schemas/v0.2/projjson.schema.json', 'type': 'Conversion', 'name': 'unknown', 'method': {'name': 'Hotine Oblique Mercator (variant B)', 'id': {'authority': 'EPSG', 'code': 9815}}, 'parameters': [{'name': 'Latitude of projection centre', 'value': latitude_projection_centre, 'unit': 'degree', 'id': {'authority': 'EPSG', 'code': 8811}}, {'name': 'Longitude of projection centre', 'value': longitude_projection_centre, 'unit': 'degree', 'id': {'authority': 'EPSG', 'code': 8812}}, {'name': 'Azimuth of initial line', 'value': azimuth_initial_line, 'unit': 'degree', 'id': {'authority': 'EPSG', 'code': 8813}}, {'name': 'Angle from Rectified to Skew Grid', 'value': angle_from_rectified_to_skew_grid, 'unit': 'degree', 'id': {'authority': 'EPSG', 'code': 8814}}, {'name': 'Scale factor on initial line', 'value': scale_factor_on_initial_line, 'unit': 'unity', 'id': {'authority': 'EPSG', 'code': 8815}}, {'name': 'Easting at projection centre', 'value': easting_projection_centre, 'unit': 'metre', 'id': {'authority': 'EPSG', 'code': 8816}}, {'name': 'Northing at projection centre', 'value': northing_projection_centre, 'unit': 'metre', 'id': {'authority': 'EPSG', 'code': 8817}}]}
        return cls.from_json_dict(omerc_json)