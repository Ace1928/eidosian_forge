from typing import Union
from pyproj._crs import CoordinateSystem
from pyproj.crs.enums import (
class Cartesian2DCS(CoordinateSystem):
    """
    .. versionadded:: 2.5.0

    This generates an Cartesian 2D Coordinate System
    """

    def __new__(cls, axis: Union[Cartesian2DCSAxis, str]=Cartesian2DCSAxis.EASTING_NORTHING):
        """
        Parameters
        ----------
        axis: :class:`pyproj.crs.enums.Cartesian2DCSAxis` or str, optional
            This is the axis order of the coordinate system.
            Default is :attr:`pyproj.crs.enums.Cartesian2DCSAxis.EASTING_NORTHING`.
        """
        return cls.from_json_dict({'type': 'CoordinateSystem', 'subtype': 'Cartesian', 'axis': _CARTESIAN_2D_AXIS_MAP[Cartesian2DCSAxis.create(axis)]})