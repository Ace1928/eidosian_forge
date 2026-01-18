from typing import Any, Callable
from functools import reduce
from collections import defaultdict
import inspect
from sympy.core.kind import Kind, UndefinedKind, NumberKind
from sympy.core.basic import Basic
from sympy.core.containers import Tuple, TupleKind
from sympy.core.decorators import sympify_method_args, sympify_return
from sympy.core.evalf import EvalfMixin
from sympy.core.expr import Expr
from sympy.core.function import Lambda
from sympy.core.logic import (FuzzyBool, fuzzy_bool, fuzzy_or, fuzzy_and,
from sympy.core.numbers import Float, Integer
from sympy.core.operations import LatticeOp
from sympy.core.parameters import global_parameters
from sympy.core.relational import Eq, Ne, is_lt
from sympy.core.singleton import Singleton, S
from sympy.core.sorting import ordered
from sympy.core.symbol import symbols, Symbol, Dummy, uniquely_named_symbol
from sympy.core.sympify import _sympify, sympify, _sympy_converter
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.miscellaneous import Max, Min
from sympy.logic.boolalg import And, Or, Not, Xor, true, false
from sympy.utilities.decorator import deprecated
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import (iproduct, sift, roundrobin, iterable,
from sympy.utilities.misc import func_name, filldedent
from mpmath import mpi, mpf
from mpmath.libmp.libmpf import prec_to_dps
class FiniteSet(Set):
    """
    Represents a finite set of Sympy expressions.

    Examples
    ========

    >>> from sympy import FiniteSet, Symbol, Interval, Naturals0
    >>> FiniteSet(1, 2, 3, 4)
    {1, 2, 3, 4}
    >>> 3 in FiniteSet(1, 2, 3, 4)
    True
    >>> FiniteSet(1, (1, 2), Symbol('x'))
    {1, x, (1, 2)}
    >>> FiniteSet(Interval(1, 2), Naturals0, {1, 2})
    FiniteSet({1, 2}, Interval(1, 2), Naturals0)
    >>> members = [1, 2, 3, 4]
    >>> f = FiniteSet(*members)
    >>> f
    {1, 2, 3, 4}
    >>> f - FiniteSet(2)
    {1, 3, 4}
    >>> f + FiniteSet(2, 5)
    {1, 2, 3, 4, 5}

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Finite_set
    """
    is_FiniteSet = True
    is_iterable = True
    is_empty = False
    is_finite_set = True

    def __new__(cls, *args, **kwargs):
        evaluate = kwargs.get('evaluate', global_parameters.evaluate)
        if evaluate:
            args = list(map(sympify, args))
            if len(args) == 0:
                return S.EmptySet
        else:
            args = list(map(sympify, args))
        dargs = {}
        for i in reversed(list(ordered(args))):
            if i.is_Symbol:
                dargs[i] = i
            else:
                try:
                    dargs[i.as_dummy()] = i
                except TypeError:
                    dargs[i] = i
        _args_set = set(dargs.values())
        args = list(ordered(_args_set, Set._infimum_key))
        obj = Basic.__new__(cls, *args)
        obj._args_set = _args_set
        return obj

    def __iter__(self):
        return iter(self.args)

    def _complement(self, other):
        if isinstance(other, Interval):
            nums, syms = ([], [])
            for m in self.args:
                if m.is_number and m.is_real:
                    nums.append(m)
                elif m.is_real == False:
                    pass
                else:
                    syms.append(m)
            if other == S.Reals and nums != []:
                nums.sort()
                intervals = []
                intervals += [Interval(S.NegativeInfinity, nums[0], True, True)]
                for a, b in zip(nums[:-1], nums[1:]):
                    intervals.append(Interval(a, b, True, True))
                intervals.append(Interval(nums[-1], S.Infinity, True, True))
                if syms != []:
                    return Complement(Union(*intervals, evaluate=False), FiniteSet(*syms), evaluate=False)
                else:
                    return Union(*intervals, evaluate=False)
            elif nums == []:
                if syms:
                    return Complement(other, FiniteSet(*syms), evaluate=False)
                else:
                    return other
        elif isinstance(other, FiniteSet):
            unk = []
            for i in self:
                c = sympify(other.contains(i))
                if c is not S.true and c is not S.false:
                    unk.append(i)
            unk = FiniteSet(*unk)
            if unk == self:
                return
            not_true = []
            for i in other:
                c = sympify(self.contains(i))
                if c is not S.true:
                    not_true.append(i)
            return Complement(FiniteSet(*not_true), unk)
        return Set._complement(self, other)

    def _contains(self, other):
        """
        Tests whether an element, other, is in the set.

        Explanation
        ===========

        The actual test is for mathematical equality (as opposed to
        syntactical equality). In the worst case all elements of the
        set must be checked.

        Examples
        ========

        >>> from sympy import FiniteSet
        >>> 1 in FiniteSet(1, 2)
        True
        >>> 5 in FiniteSet(1, 2)
        False

        """
        if other in self._args_set:
            return True
        else:
            return fuzzy_or((fuzzy_bool(Eq(e, other, evaluate=True)) for e in self.args))

    def _eval_is_subset(self, other):
        return fuzzy_and((other._contains(e) for e in self.args))

    @property
    def _boundary(self):
        return self

    @property
    def _inf(self):
        return Min(*self)

    @property
    def _sup(self):
        return Max(*self)

    @property
    def measure(self):
        return 0

    def _kind(self):
        if not self.args:
            return SetKind()
        elif all((i.kind == self.args[0].kind for i in self.args)):
            return SetKind(self.args[0].kind)
        else:
            return SetKind(UndefinedKind)

    def __len__(self):
        return len(self.args)

    def as_relational(self, symbol):
        """Rewrite a FiniteSet in terms of equalities and logic operators. """
        return Or(*[Eq(symbol, elem) for elem in self])

    def compare(self, other):
        return hash(self) - hash(other)

    def _eval_evalf(self, prec):
        dps = prec_to_dps(prec)
        return FiniteSet(*[elem.evalf(n=dps) for elem in self])

    def _eval_simplify(self, **kwargs):
        from sympy.simplify import simplify
        return FiniteSet(*[simplify(elem, **kwargs) for elem in self])

    @property
    def _sorted_args(self):
        return self.args

    def _eval_powerset(self):
        return self.func(*[self.func(*s) for s in subsets(self.args)])

    def _eval_rewrite_as_PowerSet(self, *args, **kwargs):
        """Rewriting method for a finite set to a power set."""
        from .powerset import PowerSet
        is2pow = lambda n: bool(n and (not n & n - 1))
        if not is2pow(len(self)):
            return None
        fs_test = lambda arg: isinstance(arg, Set) and arg.is_FiniteSet
        if not all((fs_test(arg) for arg in args)):
            return None
        biggest = max(args, key=len)
        for arg in subsets(biggest.args):
            arg_set = FiniteSet(*arg)
            if arg_set not in args:
                return None
        return PowerSet(biggest)

    def __ge__(self, other):
        if not isinstance(other, Set):
            raise TypeError('Invalid comparison of set with %s' % func_name(other))
        return other.is_subset(self)

    def __gt__(self, other):
        if not isinstance(other, Set):
            raise TypeError('Invalid comparison of set with %s' % func_name(other))
        return self.is_proper_superset(other)

    def __le__(self, other):
        if not isinstance(other, Set):
            raise TypeError('Invalid comparison of set with %s' % func_name(other))
        return self.is_subset(other)

    def __lt__(self, other):
        if not isinstance(other, Set):
            raise TypeError('Invalid comparison of set with %s' % func_name(other))
        return self.is_proper_subset(other)

    def __eq__(self, other):
        if isinstance(other, (set, frozenset)):
            return self._args_set == other
        return super().__eq__(other)
    __hash__: Callable[[Basic], Any] = Basic.__hash__