import itertools
import logging
import locale
import math
from numbers import Integral
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
from matplotlib import transforms as mtransforms
class LogFormatterMathtext(LogFormatter):
    """
    Format values for log axis using ``exponent = log_base(value)``.
    """

    def _non_decade_format(self, sign_string, base, fx, usetex):
        """Return string for non-decade locations."""
        return '$\\mathdefault{%s%s^{%.2f}}$' % (sign_string, base, fx)

    def __call__(self, x, pos=None):
        if x == 0:
            return '$\\mathdefault{0}$'
        sign_string = '-' if x < 0 else ''
        x = abs(x)
        b = self._base
        fx = math.log(x) / math.log(b)
        is_x_decade = _is_close_to_int(fx)
        exponent = round(fx) if is_x_decade else np.floor(fx)
        coeff = round(b ** (fx - exponent))
        if self.labelOnlyBase and (not is_x_decade):
            return ''
        if self._sublabels is not None and coeff not in self._sublabels:
            return ''
        if is_x_decade:
            fx = round(fx)
        if b % 1 == 0.0:
            base = '%d' % b
        else:
            base = '%s' % b
        if abs(fx) < mpl.rcParams['axes.formatter.min_exponent']:
            return '$\\mathdefault{%s%g}$' % (sign_string, x)
        elif not is_x_decade:
            usetex = mpl.rcParams['text.usetex']
            return self._non_decade_format(sign_string, base, fx, usetex)
        else:
            return '$\\mathdefault{%s%s^{%d}}$' % (sign_string, base, fx)