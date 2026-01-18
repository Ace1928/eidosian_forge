import math
from itertools import chain
from operator import add, mul, truediv, sub, pow
from .pyutil import defaultkeydict, deprecated
from ._expr_deprecated import _mk_PiecewisePoly, _mk_Poly  # noqa

        Parameters
        ----------
        variables : dict
        backend : module
        default : callable
            Format string based on missing key, signature: str -> str.

        Examples
        --------
        >>> def pressure(args, *params, **kw):
        ...     return args[0]*params[0]*params[1]/params[2]
        >>> Pressure = Expr.from_callback(pressure, parameter_keys='R temp vol'.split(), nargs=1)
        >>> p = Pressure([7])
        >>> p.latex({'R': 'R', 'temp': 'T', 'vol': 'V'})  # doctest: +SKIP
        '\\frac{7 R T}{V}'

        Notes
        -----
        Requires SymPy

        