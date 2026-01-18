import numpy as np
import numpy.core.numeric as nx
from numpy.compat import asbytes, asunicode
def _is_string_like(obj):
    """
    Check whether obj behaves like a string.
    """
    try:
        obj + ''
    except (TypeError, ValueError):
        return False
    return True