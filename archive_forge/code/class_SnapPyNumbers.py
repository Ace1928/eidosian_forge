from .sage_helper import _within_sage
from .pari import *
import re
class SnapPyNumbers:
    """
        Surrogate parent for a SnapPy Number, to make calls to Number.parent() work
        in or out of Sage.  This allows the following paradigm to work:

        >>> from snappy.number import Number
        >>> x = Number(1.5, precision=200)
        >>> x
        1.500000000000000000000000000000000000000000000000000000000000
        >>> y = x.parent()(2.5)
        >>> y
        2.500000000000000000000000000000000000000000000000000000000000

        >>> Number("1.0e20", precision = 53)
        1.000000000000000 E20
        >>> Number("1.0e20", precision = 53, accuracy = 0)
        1.0 E20
        >>> Number("1.0e20", precision = 53, accuracy = 3)
        1.000 E20
        >>> Number("1.23456", precision = 53)
        1.234560000000000
        >>> Number("1.23456", precision = 53, accuracy = 0)
        1.23456
        >>> Number("1.23456", precision = 53, accuracy = 4)
        1.2346
        >>> Number("0.01", precision = 53)
        0.010000000000000
        >>> Number("0.01", precision = 53, accuracy = 0)
        0.01
        >>> Number("0.01", precision = 53, accuracy = 3)
        0.010
        >>> Number("3.0123e-20", precision = 53)
        3.01230000000000 E-20
        >>> Number("3.0123e-20", precision = 53, accuracy = 0)
        3.0123 E-20
        >>> Number("3.0123e-20", precision = 53, accuracy = 3)
        3.01 E-20

        """
    _cache = {}

    def __new__(cls, precision=53):
        if precision not in SnapPyNumbers._cache:
            obj = super(SnapPyNumbers, cls).__new__(cls)
            obj._precision = precision
            SnapPyNumbers._cache[precision] = obj
            return obj
        else:
            return SnapPyNumbers._cache[precision]

    def __repr__(self):
        return 'SnapPy Numbers with %s bits precision' % self._precision

    def __call__(self, x):
        return Number(x, precision=self._precision)

    def precision(self):
        return self._precision
    prec = precision

    def pi(self):
        return self(pari.pi(precision=self._precision))

    def I(self):
        return self(pari('I'))

    def random_element(self, min=-1, max=1):
        min = self(min)
        max = self(max)
        limit = (max - min) * self(2) ** self._precision
        normalizer = self(2.0) ** (-self._precision)
        return min + normalizer * gen.random(limit.gen)