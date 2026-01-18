import ast
import re
import sys
import warnings
from ..exceptions import DTypePromotionError
from .multiarray import dtype, array, ndarray, promote_types
def _dtype_from_pep3118(spec):
    stream = _Stream(spec)
    dtype, align = __dtype_from_pep3118(stream, is_subdtype=False)
    return dtype