import numpy as np
import numpy.linalg as la
from numpy.core.multiarray import normalize_axis_index
from . import polyutils as pu
from ._polybase import ABCPolyBase
class Legendre(ABCPolyBase):
    """A Legendre series class.

    The Legendre class provides the standard Python numerical methods
    '+', '-', '*', '//', '%', 'divmod', '**', and '()' as well as the
    attributes and methods listed in the `ABCPolyBase` documentation.

    Parameters
    ----------
    coef : array_like
        Legendre coefficients in order of increasing degree, i.e.,
        ``(1, 2, 3)`` gives ``1*P_0(x) + 2*P_1(x) + 3*P_2(x)``.
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
    _add = staticmethod(legadd)
    _sub = staticmethod(legsub)
    _mul = staticmethod(legmul)
    _div = staticmethod(legdiv)
    _pow = staticmethod(legpow)
    _val = staticmethod(legval)
    _int = staticmethod(legint)
    _der = staticmethod(legder)
    _fit = staticmethod(legfit)
    _line = staticmethod(legline)
    _roots = staticmethod(legroots)
    _fromroots = staticmethod(legfromroots)
    domain = np.array(legdomain)
    window = np.array(legdomain)
    basis_name = 'P'