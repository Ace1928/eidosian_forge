import numpy as np
from warnings import warn
from ._sputils import isintlike
def _maybe_bool_ndarray(idx):
    """Returns a compatible array if elements are boolean.
    """
    idx = np.asanyarray(idx)
    if idx.dtype.kind == 'b':
        return idx
    return None