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
class Equivalent(BooleanFunction):
    """
    Equivalence relation.

    ``Equivalent(A, B)`` is True iff A and B are both True or both False.

    Returns True if all of the arguments are logically equivalent.
    Returns False otherwise.

    For two arguments, this is equivalent to :py:class:`~.Xnor`.

    Examples
    ========

    >>> from sympy.logic.boolalg import Equivalent, And
    >>> from sympy.abc import x
    >>> Equivalent(False, False, False)
    True
    >>> Equivalent(True, False, False)
    False
    >>> Equivalent(x, And(x, True))
    True

    """

    def __new__(cls, *args, **options):
        from sympy.core.relational import Relational
        args = [_sympify(arg) for arg in args]
        argset = set(args)
        for x in args:
            if isinstance(x, Number) or x in [True, False]:
                argset.discard(x)
                argset.add(bool(x))
        rel = []
        for r in argset:
            if isinstance(r, Relational):
                rel.append((r, r.canonical, r.negated.canonical))
        remove = []
        for i, (r, c, nc) in enumerate(rel):
            for j in range(i + 1, len(rel)):
                rj, cj = rel[j][:2]
                if cj == nc:
                    return false
                elif cj == c:
                    remove.append((r, rj))
                    break
        for a, b in remove:
            argset.remove(a)
            argset.remove(b)
            argset.add(True)
        if len(argset) <= 1:
            return true
        if True in argset:
            argset.discard(True)
            return And(*argset)
        if False in argset:
            argset.discard(False)
            return And(*[Not(arg) for arg in argset])
        _args = frozenset(argset)
        obj = super().__new__(cls, _args)
        obj._argset = _args
        return obj

    @property
    @cacheit
    def args(self):
        return tuple(ordered(self._argset))

    def to_nnf(self, simplify=True):
        args = []
        for a, b in zip(self.args, self.args[1:]):
            args.append(Or(Not(a), b))
        args.append(Or(Not(self.args[-1]), self.args[0]))
        return And._to_nnf(*args, simplify=simplify)

    def to_anf(self, deep=True):
        a = And(*self.args)
        b = And(*[to_anf(Not(arg), deep=False) for arg in self.args])
        b = distribute_xor_over_and(b)
        return Xor._to_anf(a, b, deep=deep)