from __future__ import annotations
import sys
import math
def _is_numpy_array(x):
    if 'numpy' not in sys.modules:
        return False
    import numpy as np
    return isinstance(x, (np.ndarray, np.generic))