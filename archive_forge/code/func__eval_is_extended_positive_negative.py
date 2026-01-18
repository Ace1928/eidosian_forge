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
def _eval_is_extended_positive_negative(self, positive):
    from sympy.polys.numberfields import minimal_polynomial
    from sympy.polys.polyerrors import NotAlgebraic
    if self.is_number:
        try:
            n2 = self._eval_evalf(2)
        except ValueError:
            return None
        if n2 is None:
            return None
        if getattr(n2, '_prec', 1) == 1:
            return None
        if n2 is S.NaN:
            return None
        f = self.evalf(2)
        if f.is_Float:
            match = (f, S.Zero)
        else:
            match = pure_complex(f)
        if match is None:
            return False
        r, i = match
        if not (i.is_Number and r.is_Number):
            return False
        if r._prec != 1 and i._prec != 1:
            return bool(not i and (r > 0 if positive else r < 0))
        elif r._prec == 1 and (not i or i._prec == 1) and self._eval_is_algebraic() and (not self.has(Function)):
            try:
                if minimal_polynomial(self).is_Symbol:
                    return False
            except (NotAlgebraic, NotImplementedError):
                pass