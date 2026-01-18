import ast
import re
import sys
import warnings
from ..exceptions import DTypePromotionError
from .multiarray import dtype, array, ndarray, promote_types
class _missing_ctypes:

    def cast(self, num, obj):
        return num.value

    class c_void_p:

        def __init__(self, ptr):
            self.value = ptr