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
@staticmethod
def _from_mpmath(x, prec):
    if hasattr(x, '_mpf_'):
        return Float._new(x._mpf_, prec)
    elif hasattr(x, '_mpc_'):
        re, im = x._mpc_
        re = Float._new(re, prec)
        im = Float._new(im, prec) * S.ImaginaryUnit
        return re + im
    else:
        raise TypeError('expected mpmath number (mpf or mpc)')