import numpy as np
import numpy.core.numeric as nx
from numpy.compat import asbytes, asunicode
@classmethod
def _find_map_entry(cls, dtype):
    for i, (deftype, func, default_def) in enumerate(cls._mapper):
        if dtype.type == deftype:
            return (i, (deftype, func, default_def))
    for i, (deftype, func, default_def) in enumerate(cls._mapper):
        if np.issubdtype(dtype.type, deftype):
            return (i, (deftype, func, default_def))
    raise LookupError