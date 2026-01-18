from __future__ import annotations
from typing import Any
from sympy.external import import_module
from sympy.printing.printer import Printer
from sympy.utilities.iterables import is_sequence
import sympy
from functools import partial
from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.exceptions import sympy_deprecation_warning
def _get_or_create(self, s, name=None, dtype=None, broadcastable=None):
    """
        Get the Theano variable for a SymPy symbol from the cache, or create it
        if it does not exist.
        """
    if name is None:
        name = s.name
    if dtype is None:
        dtype = 'floatX'
    if broadcastable is None:
        broadcastable = ()
    key = self._get_key(s, name, dtype=dtype, broadcastable=broadcastable)
    if key in self.cache:
        return self.cache[key]
    value = tt.tensor(name=name, dtype=dtype, broadcastable=broadcastable)
    self.cache[key] = value
    return value