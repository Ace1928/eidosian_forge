import operator
import numpy as np
from . import markup
from .registry import unit_registry
from .decorators import memoize
def _d_arctan2(q1, q2, out=None):
    try:
        assert q1._dimensionality == q2._dimensionality
        return Dimensionality()
    except AssertionError:
        raise ValueError('quantities must have identical units, got "%s" and "%s"' % (q1.units, q2.units))