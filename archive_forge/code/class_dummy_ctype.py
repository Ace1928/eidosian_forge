import ast
import re
import sys
import warnings
from ..exceptions import DTypePromotionError
from .multiarray import dtype, array, ndarray, promote_types
class dummy_ctype:

    def __init__(self, cls):
        self._cls = cls

    def __mul__(self, other):
        return self

    def __call__(self, *other):
        return self._cls(other)

    def __eq__(self, other):
        return self._cls == other._cls

    def __ne__(self, other):
        return self._cls != other._cls