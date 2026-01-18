import warnings
from pyproj._crs import Datum, Ellipsoid, PrimeMeridian
from pyproj.crs.coordinate_operation import (
from pyproj.crs.datum import CustomDatum, CustomEllipsoid, CustomPrimeMeridian
from pyproj.exceptions import CRSError

    http://cfconventions.org/cf-conventions/cf-conventions.html#_rotated_pole

    https://github.com/OSGeo/PROJ/pull/2835
    