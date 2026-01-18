import numpy as np
import numpy.linalg as la
from numpy.core.multiarray import normalize_axis_index
from . import polyutils as pu
from ._polybase import ABCPolyBase
class Chebyshev(ABCPolyBase):
    """A Chebyshev series class.

    The Chebyshev class provides the standard Python numerical methods
    '+', '-', '*', '//', '%', 'divmod', '**', and '()' as well as the
    methods listed below.

    Parameters
    ----------
    coef : array_like
        Chebyshev coefficients in order of increasing degree, i.e.,
        ``(1, 2, 3)`` gives ``1*T_0(x) + 2*T_1(x) + 3*T_2(x)``.
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
    _add = staticmethod(chebadd)
    _sub = staticmethod(chebsub)
    _mul = staticmethod(chebmul)
    _div = staticmethod(chebdiv)
    _pow = staticmethod(chebpow)
    _val = staticmethod(chebval)
    _int = staticmethod(chebint)
    _der = staticmethod(chebder)
    _fit = staticmethod(chebfit)
    _line = staticmethod(chebline)
    _roots = staticmethod(chebroots)
    _fromroots = staticmethod(chebfromroots)

    @classmethod
    def interpolate(cls, func, deg, domain=None, args=()):
        """Interpolate a function at the Chebyshev points of the first kind.

        Returns the series that interpolates `func` at the Chebyshev points of
        the first kind scaled and shifted to the `domain`. The resulting series
        tends to a minmax approximation of `func` when the function is
        continuous in the domain.

        .. versionadded:: 1.14.0

        Parameters
        ----------
        func : function
            The function to be interpolated. It must be a function of a single
            variable of the form ``f(x, a, b, c...)``, where ``a, b, c...`` are
            extra arguments passed in the `args` parameter.
        deg : int
            Degree of the interpolating polynomial.
        domain : {None, [beg, end]}, optional
            Domain over which `func` is interpolated. The default is None, in
            which case the domain is [-1, 1].
        args : tuple, optional
            Extra arguments to be used in the function call. Default is no
            extra arguments.

        Returns
        -------
        polynomial : Chebyshev instance
            Interpolating Chebyshev instance.

        Notes
        -----
        See `numpy.polynomial.chebfromfunction` for more details.

        """
        if domain is None:
            domain = cls.domain
        xfunc = lambda x: func(pu.mapdomain(x, cls.window, domain), *args)
        coef = chebinterpolate(xfunc, deg)
        return cls(coef, domain=domain)
    domain = np.array(chebdomain)
    window = np.array(chebdomain)
    basis_name = 'T'