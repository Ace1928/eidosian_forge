from sympy.core import (S, Add, Mul, Pow, Eq, Expr,
from sympy.core.exprtools import decompose_power, decompose_power_rat
from sympy.core.numbers import _illegal
from sympy.polys.polyerrors import PolynomialError, GeneratorsError
from sympy.polys.polyoptions import build_options
import re
class IntegerPowerable:
    """
    Mixin class for classes that define a `__mul__` method, and want to be
    raised to integer powers in the natural way that follows. Implements
    powering via binary expansion, for efficiency.

    By default, only integer powers $\\geq 2$ are supported. To support the
    first, zeroth, or negative powers, override the corresponding methods,
    `_first_power`, `_zeroth_power`, `_negative_power`, below.
    """

    def __pow__(self, e, modulo=None):
        if e < 2:
            try:
                if e == 1:
                    return self._first_power()
                elif e == 0:
                    return self._zeroth_power()
                else:
                    return self._negative_power(e, modulo=modulo)
            except NotImplementedError:
                return NotImplemented
        else:
            bits = [int(d) for d in reversed(bin(e)[2:])]
            n = len(bits)
            p = self
            first = True
            for i in range(n):
                if bits[i]:
                    if first:
                        r = p
                        first = False
                    else:
                        r *= p
                        if modulo is not None:
                            r %= modulo
                if i < n - 1:
                    p *= p
                    if modulo is not None:
                        p %= modulo
            return r

    def _negative_power(self, e, modulo=None):
        """
        Compute inverse of self, then raise that to the abs(e) power.
        For example, if the class has an `inv()` method,
            return self.inv() ** abs(e) % modulo
        """
        raise NotImplementedError

    def _zeroth_power(self):
        """Return unity element of algebraic struct to which self belongs."""
        raise NotImplementedError

    def _first_power(self):
        """Return a copy of self."""
        raise NotImplementedError