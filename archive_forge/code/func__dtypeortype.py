import numpy as np
import numpy.core.numeric as nx
from numpy.compat import asbytes, asunicode
@classmethod
def _dtypeortype(cls, dtype):
    """Returns dtype for datetime64 and type of dtype otherwise."""
    if dtype.type == np.datetime64:
        return dtype
    return dtype.type