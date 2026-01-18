import warnings
from typing import Any
from pyproj._crs import CoordinateOperation
from pyproj.exceptions import CRSError
class LambertCylindricalEqualAreaScaleConversion(CoordinateOperation):
    """
    .. versionadded:: 2.5.0

    Class for constructing the Lambert Cylindrical Equal Area conversion.

    This version uses the scale factor and differs from the official version.

    The scale factor will be converted to the Latitude of 1st standard parallel (lat_ts)
    when exporting to WKT in PROJ>=7.0.0. Previous version will export it as a
    PROJ-based coordinate operation in the WKT.

    :ref:`PROJ docs <cea>`
    """

    def __new__(cls, longitude_natural_origin: float=0.0, false_easting: float=0.0, false_northing: float=0.0, scale_factor_natural_origin: float=1.0):
        """
        Parameters
        ----------
        longitude_natural_origin: float, default=0.0
            Longitude of projection center (lon_0).
        false_easting: float, default=0.0
            False easting (x_0).
        false_northing: float, default=0.0
            False northing (y_0).
        scale_factor_natural_origin: float, default=1.0
            Scale factor at natural origin (k or k_0).

        """
        from pyproj.crs import CRS
        proj_string = f'+proj=cea +lon_0={longitude_natural_origin} +x_0={false_easting} +y_0={false_northing} +k_0={scale_factor_natural_origin}'
        return cls.from_json(CRS(proj_string).coordinate_operation.to_json())