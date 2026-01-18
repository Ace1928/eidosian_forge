from typing import Tuple as tTuple
from sympy.concrete.expr_with_limits import AddWithLimits
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.exprtools import factor_terms
from sympy.core.function import diff
from sympy.core.logic import fuzzy_bool
from sympy.core.mul import Mul
from sympy.core.numbers import oo, pi
from sympy.core.relational import Ne
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol, Wild)
from sympy.core.sympify import sympify
from sympy.functions import Piecewise, sqrt, piecewise_fold, tan, cot, atan
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.complexes import Abs, sign
from sympy.functions.elementary.miscellaneous import Min, Max
from .rationaltools import ratint
from sympy.matrices import MatrixBase
from sympy.polys import Poly, PolynomialError
from sympy.series.formal import FormalPowerSeries
from sympy.series.limits import limit
from sympy.series.order import Order
from sympy.tensor.functions import shape
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import is_sequence
from sympy.utilities.misc import filldedent
from .deltafunctions import deltaintegrate
from .meijerint import meijerint_definite, meijerint_indefinite, _debug
from .trigonometry import trigintegrate
def _calc_limit_1(F, a, b):
    """
            replace d with a, using subs if possible, otherwise limit
            where sign of b is considered
            """
    wok = F.subs(d, a)
    if wok is S.NaN or (wok.is_finite is False and a.is_finite):
        return limit(sign(b) * F, d, a)
    return wok