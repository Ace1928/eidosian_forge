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
class Subs(Expr):
    """
    Represents unevaluated substitutions of an expression.

    ``Subs(expr, x, x0)`` represents the expression resulting
    from substituting x with x0 in expr.

    Parameters
    ==========

    expr : Expr
        An expression.

    x : tuple, variable
        A variable or list of distinct variables.

    x0 : tuple or list of tuples
        A point or list of evaluation points
        corresponding to those variables.

    Examples
    ========

    >>> from sympy import Subs, Function, sin, cos
    >>> from sympy.abc import x, y, z
    >>> f = Function('f')

    Subs are created when a particular substitution cannot be made. The
    x in the derivative cannot be replaced with 0 because 0 is not a
    valid variables of differentiation:

    >>> f(x).diff(x).subs(x, 0)
    Subs(Derivative(f(x), x), x, 0)

    Once f is known, the derivative and evaluation at 0 can be done:

    >>> _.subs(f, sin).doit() == sin(x).diff(x).subs(x, 0) == cos(0)
    True

    Subs can also be created directly with one or more variables:

    >>> Subs(f(x)*sin(y) + z, (x, y), (0, 1))
    Subs(z + f(x)*sin(y), (x, y), (0, 1))
    >>> _.doit()
    z + f(0)*sin(1)

    Notes
    =====

    ``Subs`` objects are generally useful to represent unevaluated derivatives
    calculated at a point.

    The variables may be expressions, but they are subjected to the limitations
    of subs(), so it is usually a good practice to use only symbols for
    variables, since in that case there can be no ambiguity.

    There's no automatic expansion - use the method .doit() to effect all
    possible substitutions of the object and also of objects inside the
    expression.

    When evaluating derivatives at a point that is not a symbol, a Subs object
    is returned. One is also able to calculate derivatives of Subs objects - in
    this case the expression is always expanded (for the unevaluated form, use
    Derivative()).

    In order to allow expressions to combine before doit is done, a
    representation of the Subs expression is used internally to make
    expressions that are superficially different compare the same:

    >>> a, b = Subs(x, x, 0), Subs(y, y, 0)
    >>> a + b
    2*Subs(x, x, 0)

    This can lead to unexpected consequences when using methods
    like `has` that are cached:

    >>> s = Subs(x, x, 0)
    >>> s.has(x), s.has(y)
    (True, False)
    >>> ss = s.subs(x, y)
    >>> ss.has(x), ss.has(y)
    (True, False)
    >>> s, ss
    (Subs(x, x, 0), Subs(y, y, 0))
    """

    def __new__(cls, expr, variables, point, **assumptions):
        if not is_sequence(variables, Tuple):
            variables = [variables]
        variables = Tuple(*variables)
        if has_dups(variables):
            repeated = [str(v) for v, i in Counter(variables).items() if i > 1]
            __ = ', '.join(repeated)
            raise ValueError(filldedent('\n                The following expressions appear more than once: %s\n                ' % __))
        point = Tuple(*(point if is_sequence(point, Tuple) else [point]))
        if len(point) != len(variables):
            raise ValueError('Number of point values must be the same as the number of variables.')
        if not point:
            return sympify(expr)
        if isinstance(expr, Subs):
            variables = expr.variables + variables
            point = expr.point + point
            expr = expr.expr
        else:
            expr = sympify(expr)
        pre = '_'
        pts = sorted(set(point), key=default_sort_key)
        from sympy.printing.str import StrPrinter

        class CustomStrPrinter(StrPrinter):

            def _print_Dummy(self, expr):
                return str(expr) + str(expr.dummy_index)

        def mystr(expr, **settings):
            p = CustomStrPrinter(settings)
            return p.doprint(expr)
        while 1:
            s_pts = {p: Symbol(pre + mystr(p)) for p in pts}
            reps = [(v, s_pts[p]) for v, p in zip(variables, point)]
            if any((r in expr.free_symbols and r in variables and (Symbol(pre + mystr(point[variables.index(r)])) != r) for _, r in reps)):
                pre += '_'
                continue
            break
        obj = Expr.__new__(cls, expr, Tuple(*variables), point)
        obj._expr = expr.xreplace(dict(reps))
        return obj

    def _eval_is_commutative(self):
        return self.expr.is_commutative

    def doit(self, **hints):
        e, v, p = self.args
        for i, (vi, pi) in enumerate(zip(v, p)):
            if vi == pi:
                v = v[:i] + v[i + 1:]
                p = p[:i] + p[i + 1:]
        if not v:
            return self.expr
        if isinstance(e, Derivative):
            undone = []
            for i, vi in enumerate(v):
                if isinstance(vi, FunctionClass):
                    e = e.subs(vi, p[i])
                else:
                    undone.append((vi, p[i]))
            if not isinstance(e, Derivative):
                e = e.doit()
            if isinstance(e, Derivative):
                undone2 = []
                D = Dummy()
                arg = e.args[0]
                for vi, pi in undone:
                    if D not in e.xreplace({vi: D}).free_symbols:
                        if arg.has(vi):
                            e = e.subs(vi, pi)
                    else:
                        undone2.append((vi, pi))
                undone = undone2
                wrt = []
                D = Dummy()
                expr = e.expr
                free = expr.free_symbols
                for vi, ci in e.variable_count:
                    if isinstance(vi, Symbol) and vi in free:
                        expr = expr.diff((vi, ci))
                    elif D in expr.subs(vi, D).free_symbols:
                        expr = expr.diff((vi, ci))
                    else:
                        wrt.append((vi, ci))
                rv = expr.subs(undone)
                for vc in wrt:
                    rv = rv.diff(vc)
            else:
                rv = e.subs(undone)
        else:
            rv = e.doit(**hints).subs(list(zip(v, p)))
        if hints.get('deep', True) and rv != self:
            rv = rv.doit(**hints)
        return rv

    def evalf(self, prec=None, **options):
        return self.doit().evalf(prec, **options)
    n = evalf

    @property
    def variables(self):
        """The variables to be evaluated"""
        return self._args[1]
    bound_symbols = variables

    @property
    def expr(self):
        """The expression on which the substitution operates"""
        return self._args[0]

    @property
    def point(self):
        """The values for which the variables are to be substituted"""
        return self._args[2]

    @property
    def free_symbols(self):
        return self.expr.free_symbols - set(self.variables) | set(self.point.free_symbols)

    @property
    def expr_free_symbols(self):
        sympy_deprecation_warning('\n        The expr_free_symbols property is deprecated. Use free_symbols to get\n        the free symbols of an expression.\n        ', deprecated_since_version='1.9', active_deprecations_target='deprecated-expr-free-symbols')
        with ignore_warnings(SymPyDeprecationWarning):
            return self.expr.expr_free_symbols - set(self.variables) | set(self.point.expr_free_symbols)

    def __eq__(self, other):
        if not isinstance(other, Subs):
            return False
        return self._hashable_content() == other._hashable_content()

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        return super().__hash__()

    def _hashable_content(self):
        return (self._expr.xreplace(self.canonical_variables),) + tuple(ordered([(v, p) for v, p in zip(self.variables, self.point) if not self.expr.has(v)]))

    def _eval_subs(self, old, new):
        pt = list(self.point)
        if old in self.variables:
            if _atomic(new) == {new} and (not any((i.has(new) for i in self.args))):
                return self.xreplace({old: new})
            i = self.variables.index(old)
            for j in range(i, len(self.variables)):
                pt[j] = pt[j]._subs(old, new)
            return self.func(self.expr, self.variables, pt)
        v = [i._subs(old, new) for i in self.variables]
        if v != list(self.variables):
            return self.func(self.expr, self.variables + (old,), pt + [new])
        expr = self.expr._subs(old, new)
        pt = [i._subs(old, new) for i in self.point]
        return self.func(expr, v, pt)

    def _eval_derivative(self, s):
        f = self.expr
        vp = V, P = (self.variables, self.point)
        val = Add.fromiter((p.diff(s) * Subs(f.diff(v), *vp).doit() for v, p in zip(V, P)))
        efree = f.free_symbols
        compound = {i for i in efree if len(i.free_symbols) > 1}
        dums = {Dummy() for i in compound}
        masked = f.xreplace(dict(zip(compound, dums)))
        ifree = masked.free_symbols - dums
        free = ifree | compound
        free -= set(V)
        free |= {i for j in free & compound for i in j.free_symbols}
        if free & s.free_symbols:
            val += Subs(f.diff(s), self.variables, self.point).doit()
        return val

    def _eval_nseries(self, x, n, logx, cdir=0):
        if x in self.point:
            apos = self.point.index(x)
            other = self.variables[apos]
        else:
            other = x
        arg = self.expr.nseries(other, n=n, logx=logx)
        o = arg.getO()
        terms = Add.make_args(arg.removeO())
        rv = Add(*[self.func(a, *self.args[1:]) for a in terms])
        if o:
            rv += o.subs(other, x)
        return rv

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        if x in self.point:
            ipos = self.point.index(x)
            xvar = self.variables[ipos]
            return self.expr.as_leading_term(xvar)
        if x in self.variables:
            return self
        return self.expr.as_leading_term(x)