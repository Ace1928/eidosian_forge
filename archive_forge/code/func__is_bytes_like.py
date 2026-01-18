import numpy as np
import numpy.core.numeric as nx
from numpy.compat import asbytes, asunicode
def _is_bytes_like(obj):
    """
    Check whether obj behaves like a bytes object.
    """
    try:
        obj + b''
    except (TypeError, ValueError):
        return False
    return True