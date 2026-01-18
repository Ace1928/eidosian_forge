from __future__ import annotations
from typing import Any
from collections.abc import Iterable
from .add import Add
from .basic import Basic, _atomic
from .cache import cacheit
from .containers import Tuple, Dict
from .decorators import _sympifyit
from .evalf import pure_complex
from .expr import Expr, AtomicExpr
from .logic import fuzzy_and, fuzzy_or, fuzzy_not, FuzzyBool
from .mul import Mul
from .numbers import Rational, Float, Integer
from .operations import LatticeOp
from .parameters import global_parameters
from .rules import Transform
from .singleton import S
from .sympify import sympify, _sympify
from .sorting import default_sort_key, ordered
from sympy.utilities.exceptions import (sympy_deprecation_warning,
from sympy.utilities.iterables import (has_dups, sift, iterable,
from sympy.utilities.lambdify import MPMATH_TRANSLATIONS
from sympy.utilities.misc import as_int, filldedent, func_name
import mpmath
from mpmath.libmp.libmpf import prec_to_dps
import inspect
from collections import Counter
from .symbol import Dummy, Symbol
class UndefinedFunction(FunctionClass):
    """
    The (meta)class of undefined functions.
    """

    def __new__(mcl, name, bases=(AppliedUndef,), __dict__=None, **kwargs):
        from .symbol import _filter_assumptions
        assumptions, kwargs = _filter_assumptions(kwargs)
        if isinstance(name, Symbol):
            assumptions = name._merge(assumptions)
            name = name.name
        elif not isinstance(name, str):
            raise TypeError('expecting string or Symbol for name')
        else:
            commutative = assumptions.get('commutative', None)
            assumptions = Symbol(name, **assumptions).assumptions0
            if commutative is None:
                assumptions.pop('commutative')
        __dict__ = __dict__ or {}
        __dict__.update({'is_%s' % k: v for k, v in assumptions.items()})
        __dict__.update(kwargs)
        kwargs.update(assumptions)
        __dict__.update({'_kwargs': kwargs})
        __dict__['__module__'] = None
        obj = super().__new__(mcl, name, bases, __dict__)
        obj.name = name
        obj._sage_ = _undef_sage_helper
        return obj

    def __instancecheck__(cls, instance):
        return cls in type(instance).__mro__
    _kwargs: dict[str, bool | None] = {}

    def __hash__(self):
        return hash((self.class_key(), frozenset(self._kwargs.items())))

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.class_key() == other.class_key() and (self._kwargs == other._kwargs)

    def __ne__(self, other):
        return not self == other

    @property
    def _diff_wrt(self):
        return False