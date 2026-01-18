import operator
import numpy as np
from . import markup
from .registry import unit_registry
from .decorators import memoize
def _d_check_uniform(q1, q2, out=None):
    try:
        assert q1._dimensionality == q2._dimensionality
        return q1.dimensionality
    except AssertionError:
        raise ValueError('quantities must have identical units, got "%s" and "%s"' % (q1.units, q2.units))
    except AttributeError:
        try:
            if hasattr(q1, 'dimensionality'):
                if not q1._dimensionality or not np.asarray(q2).any():
                    return q1.dimensionality
                else:
                    raise ValueError
            elif hasattr(q2, 'dimensionality'):
                if not q2._dimensionality or not np.asarray(q1).any():
                    return q2.dimensionality
                else:
                    raise ValueError
        except ValueError:
            raise ValueError('quantities must have identical units, got "%s" and "%s"' % (q1.units, q2.units))