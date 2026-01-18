from __future__ import annotations
from operator import attrgetter
from collections import defaultdict
from sympy.utilities.exceptions import sympy_deprecation_warning
from .sympify import _sympify as _sympify_, sympify
from .basic import Basic
from .cache import cacheit
from .sorting import ordered
from .logic import fuzzy_and
from .parameters import global_parameters
from sympy.utilities.iterables import sift
from sympy.multipledispatch.dispatcher import (Dispatcher,
def _matches_commutative(self, expr, repl_dict=None, old=False):
    """
        Matches Add/Mul "pattern" to an expression "expr".

        repl_dict ... a dictionary of (wild: expression) pairs, that get
                      returned with the results

        This function is the main workhorse for Add/Mul.

        Examples
        ========

        >>> from sympy import symbols, Wild, sin
        >>> a = Wild("a")
        >>> b = Wild("b")
        >>> c = Wild("c")
        >>> x, y, z = symbols("x y z")
        >>> (a+sin(b)*c)._matches_commutative(x+sin(y)*z)
        {a_: x, b_: y, c_: z}

        In the example above, "a+sin(b)*c" is the pattern, and "x+sin(y)*z" is
        the expression.

        The repl_dict contains parts that were already matched. For example
        here:

        >>> (x+sin(b)*c)._matches_commutative(x+sin(y)*z, repl_dict={a: x})
        {a_: x, b_: y, c_: z}

        the only function of the repl_dict is to return it in the
        result, e.g. if you omit it:

        >>> (x+sin(b)*c)._matches_commutative(x+sin(y)*z)
        {b_: y, c_: z}

        the "a: x" is not returned in the result, but otherwise it is
        equivalent.

        """
    from .function import _coeff_isneg
    from .expr import Expr
    if isinstance(self, Expr) and (not isinstance(expr, Expr)):
        return None
    if repl_dict is None:
        repl_dict = {}
    if self == expr:
        return repl_dict
    d = self._matches_simple(expr, repl_dict)
    if d is not None:
        return d
    from .function import WildFunction
    from .symbol import Wild
    wild_part, exact_part = sift(self.args, lambda p: p.has(Wild, WildFunction) and (not expr.has(p)), binary=True)
    if not exact_part:
        wild_part = list(ordered(wild_part))
        if self.is_Add:
            wild_part = sorted(wild_part, key=lambda x: x.args[0] if x.is_Mul and x.args[0].is_Number else 0)
    else:
        exact = self._new_rawargs(*exact_part)
        free = expr.free_symbols
        if free and exact.free_symbols - free:
            return None
        newexpr = self._combine_inverse(expr, exact)
        if not old and (expr.is_Add or expr.is_Mul):
            check = newexpr
            if _coeff_isneg(check):
                check = -check
            if check.count_ops() > expr.count_ops():
                return None
        newpattern = self._new_rawargs(*wild_part)
        return newpattern.matches(newexpr, repl_dict)
    i = 0
    saw = set()
    while expr not in saw:
        saw.add(expr)
        args = tuple(ordered(self.make_args(expr)))
        if self.is_Add and expr.is_Add:
            args = tuple(sorted(args, key=lambda x: x.args[0] if x.is_Mul and x.args[0].is_Number else 0))
        expr_list = (self.identity,) + args
        for last_op in reversed(expr_list):
            for w in reversed(wild_part):
                d1 = w.matches(last_op, repl_dict)
                if d1 is not None:
                    d2 = self.xreplace(d1).matches(expr, d1)
                    if d2 is not None:
                        return d2
        if i == 0:
            if self.is_Mul:
                if expr.is_Pow and expr.exp.is_Integer:
                    from .mul import Mul
                    if expr.exp > 0:
                        expr = Mul(*[expr.base, expr.base ** (expr.exp - 1)], evaluate=False)
                    else:
                        expr = Mul(*[1 / expr.base, expr.base ** (expr.exp + 1)], evaluate=False)
                    i += 1
                    continue
            elif self.is_Add:
                c, e = expr.as_coeff_Mul()
                if abs(c) > 1:
                    from .add import Add
                    if c > 0:
                        expr = Add(*[e, (c - 1) * e], evaluate=False)
                    else:
                        expr = Add(*[-e, (c + 1) * e], evaluate=False)
                    i += 1
                    continue
                from sympy.simplify.radsimp import collect
                was = expr
                did = set()
                for w in reversed(wild_part):
                    c, w = w.as_coeff_mul(Wild)
                    free = c.free_symbols - did
                    if free:
                        did.update(free)
                        expr = collect(expr, free)
                if expr != was:
                    i += 0
                    continue
            break
    return