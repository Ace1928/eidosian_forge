import operator
import numpy as np
from . import markup
from .registry import unit_registry
from .decorators import memoize
def _d_trig(q1, out=None):
    try:
        assert q1.units == unit_registry['radian']
    except AssertionError:
        raise ValueError('expected units of radians, got "%s"' % q1._dimensionality)
    return Dimensionality()