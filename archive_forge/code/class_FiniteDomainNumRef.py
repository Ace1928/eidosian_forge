from . import z3core
from .z3core import *
from .z3types import *
from .z3consts import *
from .z3printer import *
from fractions import Fraction
import sys
import io
import math
import copy
class FiniteDomainNumRef(FiniteDomainRef):
    """Integer values."""

    def as_long(self):
        """Return a Z3 finite-domain numeral as a Python long (bignum) numeral.

        >>> s = FiniteDomainSort('S', 100)
        >>> v = FiniteDomainVal(3, s)
        >>> v
        3
        >>> v.as_long() + 1
        4
        """
        return int(self.as_string())

    def as_string(self):
        """Return a Z3 finite-domain numeral as a Python string.

        >>> s = FiniteDomainSort('S', 100)
        >>> v = FiniteDomainVal(42, s)
        >>> v.as_string()
        '42'
        """
        return Z3_get_numeral_string(self.ctx_ref(), self.as_ast())