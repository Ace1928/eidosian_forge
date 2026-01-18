from typing import Union
from pyproj._crs import CoordinateSystem
from pyproj.crs.enums import (
class VerticalCS(CoordinateSystem):
    """
    .. versionadded:: 2.5.0

    This generates an Vertical Coordinate System
    """

    def __new__(cls, axis: Union[VerticalCSAxis, str]=VerticalCSAxis.GRAVITY_HEIGHT):
        """
        Parameters
        ----------
        axis: :class:`pyproj.crs.enums.VerticalCSAxis` or str, optional
            This is the axis direction of the coordinate system.
            Default is :attr:`pyproj.crs.enums.VerticalCSAxis.GRAVITY_HEIGHT`.
        """
        return cls.from_json_dict({'type': 'CoordinateSystem', 'subtype': 'vertical', 'axis': [_VERTICAL_AXIS_MAP[VerticalCSAxis.create(axis)]]})