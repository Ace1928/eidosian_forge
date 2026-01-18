import operator
import numpy as np
from . import markup
from .registry import unit_registry
from .decorators import memoize
def _d_radians(q1, out=None):
    try:
        assert q1.units == unit_registry['degree']
    except AssertionError:
        raise ValueError('expected units of degrees, got "%s"' % q1._dimensionality)
    return unit_registry['radian'].dimensionality