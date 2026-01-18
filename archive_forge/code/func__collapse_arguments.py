from sympy.core import Function, S, sympify, NumberKind
from sympy.utilities.iterables import sift
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.operations import LatticeOp, ShortCircuit
from sympy.core.function import (Application, Lambda,
from sympy.core.expr import Expr
from sympy.core.exprtools import factor_terms
from sympy.core.mod import Mod
from sympy.core.mul import Mul
from sympy.core.numbers import Rational
from sympy.core.power import Pow
from sympy.core.relational import Eq, Relational
from sympy.core.singleton import Singleton
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy
from sympy.core.rules import Transform
from sympy.core.logic import fuzzy_and, fuzzy_or, _torf
from sympy.core.traversal import walk
from sympy.core.numbers import Integer
from sympy.logic.boolalg import And, Or
@classmethod
def _collapse_arguments(cls, args, **assumptions):
    """Remove redundant args.

        Examples
        ========

        >>> from sympy import Min, Max
        >>> from sympy.abc import a, b, c, d, e

        Any arg in parent that appears in any
        parent-like function in any of the flat args
        of parent can be removed from that sub-arg:

        >>> Min(a, Max(b, Min(a, c, d)))
        Min(a, Max(b, Min(c, d)))

        If the arg of parent appears in an opposite-than parent
        function in any of the flat args of parent that function
        can be replaced with the arg:

        >>> Min(a, Max(b, Min(c, d, Max(a, e))))
        Min(a, Max(b, Min(a, c, d)))
        """
    if not args:
        return args
    args = list(ordered(args))
    if cls == Min:
        other = Max
    else:
        other = Min
    if args[0].is_number:
        sifted = mins, maxs = ([], [])
        for i in args:
            for v in walk(i, Min, Max):
                if v.args[0].is_comparable:
                    sifted[isinstance(v, Max)].append(v)
        small = Min.identity
        for i in mins:
            v = i.args[0]
            if v.is_number and (v < small) == True:
                small = v
        big = Max.identity
        for i in maxs:
            v = i.args[0]
            if v.is_number and (v > big) == True:
                big = v
        if cls == Min:
            for arg in args:
                if not arg.is_number:
                    break
                if (arg < small) == True:
                    small = arg
        elif cls == Max:
            for arg in args:
                if not arg.is_number:
                    break
                if (arg > big) == True:
                    big = arg
        T = None
        if cls == Min:
            if small != Min.identity:
                other = Max
                T = small
        elif big != Max.identity:
            other = Min
            T = big
        if T is not None:
            for i in range(len(args)):
                a = args[i]
                if isinstance(a, other):
                    a0 = a.args[0]
                    if (a0 > T if other == Max else a0 < T) == True:
                        args[i] = cls.identity

    def do(ai, a):
        if not isinstance(ai, (Min, Max)):
            return ai
        cond = a in ai.args
        if not cond:
            return ai.func(*[do(i, a) for i in ai.args], evaluate=False)
        if isinstance(ai, cls):
            return ai.func(*[do(i, a) for i in ai.args if i != a], evaluate=False)
        return a
    for i, a in enumerate(args):
        args[i + 1:] = [do(ai, a) for ai in args[i + 1:]]

    def factor_minmax(args):
        is_other = lambda arg: isinstance(arg, other)
        other_args, remaining_args = sift(args, is_other, binary=True)
        if not other_args:
            return args
        arg_sets = [set(arg.args) for arg in other_args]
        common = set.intersection(*arg_sets)
        if not common:
            return args
        new_other_args = list(common)
        arg_sets_diff = [arg_set - common for arg_set in arg_sets]
        if all(arg_sets_diff):
            other_args_diff = [other(*s, evaluate=False) for s in arg_sets_diff]
            new_other_args.append(cls(*other_args_diff, evaluate=False))
        other_args_factored = other(*new_other_args, evaluate=False)
        return remaining_args + [other_args_factored]
    if len(args) > 1:
        args = factor_minmax(args)
    return args