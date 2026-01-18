from __future__ import annotations
from typing import Tuple as tTuple, Optional, Union as tUnion, Callable, List, Dict as tDict, Type, TYPE_CHECKING, \
import math
import mpmath.libmp as libmp
from mpmath import (
from mpmath import inf as mpmath_inf
from mpmath.libmp import (from_int, from_man_exp, from_rational, fhalf,
from mpmath.libmp import bitcount as mpmath_bitcount
from mpmath.libmp.backend import MPZ
from mpmath.libmp.libmpc import _infs_nan
from mpmath.libmp.libmpf import dps_to_prec, prec_to_dps
from .sympify import sympify
from .singleton import S
from sympy.external.gmpy import SYMPY_INTS
from sympy.utilities.iterables import is_sequence
from sympy.utilities.lambdify import lambdify
from sympy.utilities.misc import as_int
def _evalf_with_bounded_error(x: 'Expr', eps: 'Optional[Expr]'=None, m: int=0, options: Optional[OPT_DICT]=None) -> TMP_RES:
    """
    Evaluate *x* to within a bounded absolute error.

    Parameters
    ==========

    x : Expr
        The quantity to be evaluated.
    eps : Expr, None, optional (default=None)
        Positive real upper bound on the acceptable error.
    m : int, optional (default=0)
        If *eps* is None, then use 2**(-m) as the upper bound on the error.
    options: OPT_DICT
        As in the ``evalf`` function.

    Returns
    =======

    A tuple ``(re, im, re_acc, im_acc)``, as returned by ``evalf``.

    See Also
    ========

    evalf

    """
    if eps is not None:
        if not (eps.is_Rational or eps.is_Float) or not eps > 0:
            raise ValueError('eps must be positive')
        r, _, _, _ = evalf(1 / eps, 1, {})
        m = fastlog(r)
    c, d, _, _ = evalf(x, 1, {})
    nr, ni = (fastlog(c), fastlog(d))
    n = max(nr, ni) + 1
    p = max(1, m + n + 1)
    options = options or {}
    return evalf(x, p, options)