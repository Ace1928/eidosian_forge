import numpy as np
import numpy.linalg as la
from numpy.core.multiarray import normalize_axis_index
from . import polyutils as pu
from ._polybase import ABCPolyBase
class Polynomial(ABCPolyBase):
    """A power series class.

    The Polynomial class provides the standard Python numerical methods
    '+', '-', '*', '//', '%', 'divmod', '**', and '()' as well as the
    attributes and methods listed in the `ABCPolyBase` documentation.

    Parameters
    ----------
    coef : array_like
        Polynomial coefficients in order of increasing degree, i.e.,
        ``(1, 2, 3)`` give ``1 + 2*x + 3*x**2``.
    domain : (2,) array_like, optional
        Domain to use. The interval ``[domain[0], domain[1]]`` is mapped
        to the interval ``[window[0], window[1]]`` by shifting and scaling.
        The default value is [-1, 1].
    window : (2,) array_like, optional
        Window, see `domain` for its use. The default value is [-1, 1].

        .. versionadded:: 1.6.0
    symbol : str, optional
        Symbol used to represent the independent variable in string
        representations of the polynomial expression, e.g. for printing.
        The symbol must be a valid Python identifier. Default value is 'x'.

        .. versionadded:: 1.24

    """
    _add = staticmethod(polyadd)
    _sub = staticmethod(polysub)
    _mul = staticmethod(polymul)
    _div = staticmethod(polydiv)
    _pow = staticmethod(polypow)
    _val = staticmethod(polyval)
    _int = staticmethod(polyint)
    _der = staticmethod(polyder)
    _fit = staticmethod(polyfit)
    _line = staticmethod(polyline)
    _roots = staticmethod(polyroots)
    _fromroots = staticmethod(polyfromroots)
    domain = np.array(polydomain)
    window = np.array(polydomain)
    basis_name = None

    @classmethod
    def _str_term_unicode(cls, i, arg_str):
        if i == '1':
            return f'·{arg_str}'
        else:
            return f'·{arg_str}{i.translate(cls._superscript_mapping)}'

    @staticmethod
    def _str_term_ascii(i, arg_str):
        if i == '1':
            return f' {arg_str}'
        else:
            return f' {arg_str}**{i}'

    @staticmethod
    def _repr_latex_term(i, arg_str, needs_parens):
        if needs_parens:
            arg_str = f'\\left({arg_str}\\right)'
        if i == 0:
            return '1'
        elif i == 1:
            return arg_str
        else:
            return f'{arg_str}^{{{i}}}'