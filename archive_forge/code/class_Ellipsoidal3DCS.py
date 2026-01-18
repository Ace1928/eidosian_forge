from typing import Union
from pyproj._crs import CoordinateSystem
from pyproj.crs.enums import (
class Ellipsoidal3DCS(CoordinateSystem):
    """
    .. versionadded:: 2.5.0

    This generates an Ellipsoidal 3D Coordinate System
    """

    def __new__(cls, axis: Union[Ellipsoidal3DCSAxis, str]=Ellipsoidal3DCSAxis.LONGITUDE_LATITUDE_HEIGHT):
        """
        Parameters
        ----------
        axis: :class:`pyproj.crs.enums.Ellipsoidal3DCSAxis` or str, optional
            This is the axis order of the coordinate system. Default is
            :attr:`pyproj.crs.enums.Ellipsoidal3DCSAxis.LONGITUDE_LATITUDE_HEIGHT`.
        """
        return cls.from_json_dict({'type': 'CoordinateSystem', 'subtype': 'ellipsoidal', 'axis': _ELLIPSOIDAL_3D_AXIS_MAP[Ellipsoidal3DCSAxis.create(axis)]})