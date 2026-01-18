import operator
import numpy as np
from . import markup
from .registry import unit_registry
from .decorators import memoize
def _d_power(q1, q2, out=None):
    if getattr(q2, 'dimensionality', None):
        raise ValueError('exponent must be dimensionless')
    try:
        q2 = np.array(q2)
        p = q2.min()
        if p != q2.max():
            raise ValueError('Quantities must be raised to a uniform power')
        return q1._dimensionality ** p
    except AttributeError:
        return Dimensionality()