from typing import Union
from pyproj._crs import CoordinateSystem
from pyproj.crs.enums import (

        Parameters
        ----------
        axis: :class:`pyproj.crs.enums.VerticalCSAxis` or str, optional
            This is the axis direction of the coordinate system.
            Default is :attr:`pyproj.crs.enums.VerticalCSAxis.GRAVITY_HEIGHT`.
        