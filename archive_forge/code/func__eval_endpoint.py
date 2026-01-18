from __future__ import annotations
from typing import TYPE_CHECKING
from collections.abc import Iterable
from functools import reduce
import re
from .sympify import sympify, _sympify
from .basic import Basic, Atom
from .singleton import S
from .evalf import EvalfMixin, pure_complex, DEFAULT_MAXPREC
from .decorators import call_highest_priority, sympify_method_args, sympify_return
from .cache import cacheit
from .sorting import default_sort_key
from .kind import NumberKind
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.misc import as_int, func_name, filldedent
from sympy.utilities.iterables import has_variety, sift
from mpmath.libmp import mpf_log, prec_to_dps
from mpmath.libmp.libintmath import giant_steps
from collections import defaultdict
from .mul import Mul
from .add import Add
from .power import Pow
from .function import Function, _derivative_dispatch
from .mod import Mod
from .exprtools import factor_terms
from .numbers import Float, Integer, Rational, _illegal
def _eval_endpoint(left):
    c = a if left else b
    if c is None:
        return S.Zero
    else:
        C = self.subs(x, c)
        if C.has(S.NaN, S.Infinity, S.NegativeInfinity, S.ComplexInfinity, AccumBounds):
            if (a < b) != False:
                C = limit(self, x, c, '+' if left else '-')
            else:
                C = limit(self, x, c, '-' if left else '+')
            if isinstance(C, Limit):
                raise NotImplementedError('Could not compute limit')
    return C