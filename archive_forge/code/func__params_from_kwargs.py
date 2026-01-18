import math
import warnings
from typing import Any, Optional, Union
from pyproj._geod import Geod as _Geod
from pyproj._geod import GeodIntermediateReturn, geodesic_version_str
from pyproj._geod import reverse_azimuth as _reverse_azimuth
from pyproj.enums import GeodIntermediateFlag
from pyproj.exceptions import GeodError
from pyproj.list import get_ellps_map
from pyproj.utils import DataType, _convertback, _copytobuffer
def _params_from_kwargs(kwargs: dict) -> tuple[float, float, float, float]:
    """
    Build Geodesic parameters from input kwargs:

    - a: the semi-major axis (required).

    Need least one of these parameters.

    - b: the semi-minor axis
    - rf: the reciprocal flattening
    - f: flattening
    - es: eccentricity squared


    Parameter
    ---------
    kwargs: dict
        The input kwargs for an ellipse.

    Returns
    -------
    tuple[float, float, float, float]

    """
    semi_major_axis = kwargs['a']
    if 'b' in kwargs:
        semi_minor_axis = kwargs['b']
        eccentricity_squared = 1.0 - semi_minor_axis ** 2 / semi_major_axis ** 2
        flattening = (semi_major_axis - semi_minor_axis) / semi_major_axis
    elif 'rf' in kwargs:
        flattening = 1.0 / kwargs['rf']
        semi_minor_axis = semi_major_axis * (1.0 - flattening)
        eccentricity_squared = 1.0 - semi_minor_axis ** 2 / semi_major_axis ** 2
    elif 'f' in kwargs:
        flattening = kwargs['f']
        semi_minor_axis = semi_major_axis * (1.0 - flattening)
        eccentricity_squared = 1.0 - (semi_minor_axis / semi_major_axis) ** 2
    elif 'es' in kwargs:
        eccentricity_squared = kwargs['es']
        semi_minor_axis = math.sqrt(semi_major_axis ** 2 - eccentricity_squared * semi_major_axis ** 2)
        flattening = (semi_major_axis - semi_minor_axis) / semi_major_axis
    elif 'e' in kwargs:
        eccentricity_squared = kwargs['e'] ** 2
        semi_minor_axis = math.sqrt(semi_major_axis ** 2 - eccentricity_squared * semi_major_axis ** 2)
        flattening = (semi_major_axis - semi_minor_axis) / semi_major_axis
    else:
        semi_minor_axis = semi_major_axis
        flattening = 0.0
        eccentricity_squared = 0.0
    return (semi_major_axis, semi_minor_axis, flattening, eccentricity_squared)