import warnings
from typing import Any
from pyproj._crs import CoordinateOperation
from pyproj.exceptions import CRSError

        Parameters
        ----------
        source_crs: Any
            Input to create the Source CRS.
        x_axis_translation: float, default=0.0
            X-axis translation.
        y_axis_translation: float, default=0.0
            Y-axis translation.
        z_axis_translation: float, default=0.0
            Z-axis translation.
        x_axis_rotation: float, default=0.0
            X-axis rotation.
        y_axis_rotation: float, default=0.0
            Y-axis rotation.
        z_axis_rotation: float, default=0.0
            Z-axis rotation.
        scale_difference: float, default=0.0
            Scale difference.
        