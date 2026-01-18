import numpy as np
import numpy.core.numeric as nx
from numpy.compat import asbytes, asunicode
@classmethod
def _getsubdtype(cls, val):
    """Returns the type of the dtype of the input variable."""
    return np.array(val).dtype.type