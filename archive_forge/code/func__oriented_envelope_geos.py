import numpy as np
from shapely import lib
from shapely._enum import ParamEnum
from shapely.algorithms._oriented_envelope import _oriented_envelope_min_area_vectorized
from shapely.decorators import multithreading_enabled, requires_geos
@requires_geos('3.6.0')
@multithreading_enabled
def _oriented_envelope_geos(geometry, **kwargs):
    return lib.oriented_envelope(geometry, **kwargs)