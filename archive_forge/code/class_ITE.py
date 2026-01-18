from collections import defaultdict
from itertools import chain, combinations, product, permutations
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.containers import Tuple
from sympy.core.decorators import sympify_method_args, sympify_return
from sympy.core.function import Application, Derivative
from sympy.core.kind import BooleanKind, NumberKind
from sympy.core.numbers import Number
from sympy.core.operations import LatticeOp
from sympy.core.singleton import Singleton, S
from sympy.core.sorting import ordered
from sympy.core.sympify import _sympy_converter, _sympify, sympify
from sympy.utilities.iterables import sift, ibin
from sympy.utilities.misc import filldedent
class ITE(BooleanFunction):
    """
    If-then-else clause.

    ``ITE(A, B, C)`` evaluates and returns the result of B if A is true
    else it returns the result of C. All args must be Booleans.

    From a logic gate perspective, ITE corresponds to a 2-to-1 multiplexer,
    where A is the select signal.

    Examples
    ========

    >>> from sympy.logic.boolalg import ITE, And, Xor, Or
    >>> from sympy.abc import x, y, z
    >>> ITE(True, False, True)
    False
    >>> ITE(Or(True, False), And(True, True), Xor(True, True))
    True
    >>> ITE(x, y, z)
    ITE(x, y, z)
    >>> ITE(True, x, y)
    x
    >>> ITE(False, x, y)
    y
    >>> ITE(x, y, y)
    y

    Trying to use non-Boolean args will generate a TypeError:

    >>> ITE(True, [], ())
    Traceback (most recent call last):
    ...
    TypeError: expecting bool, Boolean or ITE, not `[]`

    """

    def __new__(cls, *args, **kwargs):
        from sympy.core.relational import Eq, Ne
        if len(args) != 3:
            raise ValueError('expecting exactly 3 args')
        a, b, c = args
        if isinstance(a, (Eq, Ne)):
            b, c = map(as_Boolean, (b, c))
            bin_syms = set().union(*[i.binary_symbols for i in (b, c)])
            if len(set(a.args) - bin_syms) == 1:
                _a = a
                if a.lhs is true:
                    a = a.rhs
                elif a.rhs is true:
                    a = a.lhs
                elif a.lhs is false:
                    a = Not(a.rhs)
                elif a.rhs is false:
                    a = Not(a.lhs)
                else:
                    a = false
                if isinstance(_a, Ne):
                    a = Not(a)
        else:
            a, b, c = BooleanFunction.binary_check_and_simplify(a, b, c)
        rv = None
        if kwargs.get('evaluate', True):
            rv = cls.eval(a, b, c)
        if rv is None:
            rv = BooleanFunction.__new__(cls, a, b, c, evaluate=False)
        return rv

    @classmethod
    def eval(cls, *args):
        from sympy.core.relational import Eq, Ne
        a, b, c = args
        if isinstance(a, (Ne, Eq)):
            _a = a
            if true in a.args:
                a = a.lhs if a.rhs is true else a.rhs
            elif false in a.args:
                a = Not(a.lhs) if a.rhs is false else Not(a.rhs)
            else:
                _a = None
            if _a is not None and isinstance(_a, Ne):
                a = Not(a)
        if a is true:
            return b
        if a is false:
            return c
        if b == c:
            return b
        else:
            if b is true and c is false:
                return a
            if b is false and c is true:
                return Not(a)
        if [a, b, c] != args:
            return cls(a, b, c, evaluate=False)

    def to_nnf(self, simplify=True):
        a, b, c = self.args
        return And._to_nnf(Or(Not(a), b), Or(a, c), simplify=simplify)

    def _eval_as_set(self):
        return self.to_nnf().as_set()

    def _eval_rewrite_as_Piecewise(self, *args, **kwargs):
        from sympy.functions.elementary.piecewise import Piecewise
        return Piecewise((args[1], args[0]), (args[2], True))