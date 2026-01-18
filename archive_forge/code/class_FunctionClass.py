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
class FunctionClass(type):
    """
    Base class for function classes. FunctionClass is a subclass of type.

    Use Function('<function name>' [ , signature ]) to create
    undefined function classes.
    """
    _new = type.__new__

    def __init__(cls, *args, **kwargs):
        nargs = kwargs.pop('nargs', cls.__dict__.get('nargs', arity(cls)))
        if nargs is None and 'nargs' not in cls.__dict__:
            for supcls in cls.__mro__:
                if hasattr(supcls, '_nargs'):
                    nargs = supcls._nargs
                    break
                else:
                    continue
        if is_sequence(nargs):
            if not nargs:
                raise ValueError(filldedent('\n                    Incorrectly specified nargs as %s:\n                    if there are no arguments, it should be\n                    `nargs = 0`;\n                    if there are any number of arguments,\n                    it should be\n                    `nargs = None`' % str(nargs)))
            nargs = tuple(ordered(set(nargs)))
        elif nargs is not None:
            nargs = (as_int(nargs),)
        cls._nargs = nargs
        if len(args) == 3:
            namespace = args[2]
            if 'eval' in namespace and (not isinstance(namespace['eval'], classmethod)):
                raise TypeError('eval on Function subclasses should be a class method (defined with @classmethod)')

    @property
    def __signature__(self):
        """
        Allow Python 3's inspect.signature to give a useful signature for
        Function subclasses.
        """
        try:
            from inspect import signature
        except ImportError:
            return None
        return signature(self.eval)

    @property
    def free_symbols(self):
        return set()

    @property
    def xreplace(self):
        return lambda rule, **_: rule.get(self, self)

    @property
    def nargs(self):
        """Return a set of the allowed number of arguments for the function.

        Examples
        ========

        >>> from sympy import Function
        >>> f = Function('f')

        If the function can take any number of arguments, the set of whole
        numbers is returned:

        >>> Function('f').nargs
        Naturals0

        If the function was initialized to accept one or more arguments, a
        corresponding set will be returned:

        >>> Function('f', nargs=1).nargs
        {1}
        >>> Function('f', nargs=(2, 1)).nargs
        {1, 2}

        The undefined function, after application, also has the nargs
        attribute; the actual number of arguments is always available by
        checking the ``args`` attribute:

        >>> f = Function('f')
        >>> f(1).nargs
        Naturals0
        >>> len(f(1).args)
        1
        """
        from sympy.sets.sets import FiniteSet
        return FiniteSet(*self._nargs) if self._nargs else S.Naturals0

    def _valid_nargs(self, n: int) -> bool:
        """ Return True if the specified integer is a valid number of arguments

        The number of arguments n is guaranteed to be an integer and positive

        """
        if self._nargs:
            return n in self._nargs
        nargs = self.nargs
        return nargs is S.Naturals0 or n in nargs

    def __repr__(cls):
        return cls.__name__