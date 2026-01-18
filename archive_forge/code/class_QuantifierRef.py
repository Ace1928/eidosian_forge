from . import z3core
from .z3core import *
from .z3types import *
from .z3consts import *
from .z3printer import *
from fractions import Fraction
import sys
import io
import math
import copy
class QuantifierRef(BoolRef):
    """Universally and Existentially quantified formulas."""

    def as_ast(self):
        return self.ast

    def get_id(self):
        return Z3_get_ast_id(self.ctx_ref(), self.as_ast())

    def sort(self):
        """Return the Boolean sort or sort of Lambda."""
        if self.is_lambda():
            return _sort(self.ctx, self.as_ast())
        return BoolSort(self.ctx)

    def is_forall(self):
        """Return `True` if `self` is a universal quantifier.

        >>> f = Function('f', IntSort(), IntSort())
        >>> x = Int('x')
        >>> q = ForAll(x, f(x) == 0)
        >>> q.is_forall()
        True
        >>> q = Exists(x, f(x) != 0)
        >>> q.is_forall()
        False
        """
        return Z3_is_quantifier_forall(self.ctx_ref(), self.ast)

    def is_exists(self):
        """Return `True` if `self` is an existential quantifier.

        >>> f = Function('f', IntSort(), IntSort())
        >>> x = Int('x')
        >>> q = ForAll(x, f(x) == 0)
        >>> q.is_exists()
        False
        >>> q = Exists(x, f(x) != 0)
        >>> q.is_exists()
        True
        """
        return Z3_is_quantifier_exists(self.ctx_ref(), self.ast)

    def is_lambda(self):
        """Return `True` if `self` is a lambda expression.

        >>> f = Function('f', IntSort(), IntSort())
        >>> x = Int('x')
        >>> q = Lambda(x, f(x))
        >>> q.is_lambda()
        True
        >>> q = Exists(x, f(x) != 0)
        >>> q.is_lambda()
        False
        """
        return Z3_is_lambda(self.ctx_ref(), self.ast)

    def __getitem__(self, arg):
        """Return the Z3 expression `self[arg]`.
        """
        if z3_debug():
            _z3_assert(self.is_lambda(), 'quantifier should be a lambda expression')
        return _array_select(self, arg)

    def weight(self):
        """Return the weight annotation of `self`.

        >>> f = Function('f', IntSort(), IntSort())
        >>> x = Int('x')
        >>> q = ForAll(x, f(x) == 0)
        >>> q.weight()
        1
        >>> q = ForAll(x, f(x) == 0, weight=10)
        >>> q.weight()
        10
        """
        return int(Z3_get_quantifier_weight(self.ctx_ref(), self.ast))

    def skolem_id(self):
        """Return the skolem id of `self`.
        """
        return _symbol2py(self.ctx, Z3_get_quantifier_skolem_id(self.ctx_ref(), self.ast))

    def qid(self):
        """Return the quantifier id of `self`.
        """
        return _symbol2py(self.ctx, Z3_get_quantifier_id(self.ctx_ref(), self.ast))

    def num_patterns(self):
        """Return the number of patterns (i.e., quantifier instantiation hints) in `self`.

        >>> f = Function('f', IntSort(), IntSort())
        >>> g = Function('g', IntSort(), IntSort())
        >>> x = Int('x')
        >>> q = ForAll(x, f(x) != g(x), patterns = [ f(x), g(x) ])
        >>> q.num_patterns()
        2
        """
        return int(Z3_get_quantifier_num_patterns(self.ctx_ref(), self.ast))

    def pattern(self, idx):
        """Return a pattern (i.e., quantifier instantiation hints) in `self`.

        >>> f = Function('f', IntSort(), IntSort())
        >>> g = Function('g', IntSort(), IntSort())
        >>> x = Int('x')
        >>> q = ForAll(x, f(x) != g(x), patterns = [ f(x), g(x) ])
        >>> q.num_patterns()
        2
        >>> q.pattern(0)
        f(Var(0))
        >>> q.pattern(1)
        g(Var(0))
        """
        if z3_debug():
            _z3_assert(idx < self.num_patterns(), 'Invalid pattern idx')
        return PatternRef(Z3_get_quantifier_pattern_ast(self.ctx_ref(), self.ast, idx), self.ctx)

    def num_no_patterns(self):
        """Return the number of no-patterns."""
        return Z3_get_quantifier_num_no_patterns(self.ctx_ref(), self.ast)

    def no_pattern(self, idx):
        """Return a no-pattern."""
        if z3_debug():
            _z3_assert(idx < self.num_no_patterns(), 'Invalid no-pattern idx')
        return _to_expr_ref(Z3_get_quantifier_no_pattern_ast(self.ctx_ref(), self.ast, idx), self.ctx)

    def body(self):
        """Return the expression being quantified.

        >>> f = Function('f', IntSort(), IntSort())
        >>> x = Int('x')
        >>> q = ForAll(x, f(x) == 0)
        >>> q.body()
        f(Var(0)) == 0
        """
        return _to_expr_ref(Z3_get_quantifier_body(self.ctx_ref(), self.ast), self.ctx)

    def num_vars(self):
        """Return the number of variables bounded by this quantifier.

        >>> f = Function('f', IntSort(), IntSort(), IntSort())
        >>> x = Int('x')
        >>> y = Int('y')
        >>> q = ForAll([x, y], f(x, y) >= x)
        >>> q.num_vars()
        2
        """
        return int(Z3_get_quantifier_num_bound(self.ctx_ref(), self.ast))

    def var_name(self, idx):
        """Return a string representing a name used when displaying the quantifier.

        >>> f = Function('f', IntSort(), IntSort(), IntSort())
        >>> x = Int('x')
        >>> y = Int('y')
        >>> q = ForAll([x, y], f(x, y) >= x)
        >>> q.var_name(0)
        'x'
        >>> q.var_name(1)
        'y'
        """
        if z3_debug():
            _z3_assert(idx < self.num_vars(), 'Invalid variable idx')
        return _symbol2py(self.ctx, Z3_get_quantifier_bound_name(self.ctx_ref(), self.ast, idx))

    def var_sort(self, idx):
        """Return the sort of a bound variable.

        >>> f = Function('f', IntSort(), RealSort(), IntSort())
        >>> x = Int('x')
        >>> y = Real('y')
        >>> q = ForAll([x, y], f(x, y) >= x)
        >>> q.var_sort(0)
        Int
        >>> q.var_sort(1)
        Real
        """
        if z3_debug():
            _z3_assert(idx < self.num_vars(), 'Invalid variable idx')
        return _to_sort_ref(Z3_get_quantifier_bound_sort(self.ctx_ref(), self.ast, idx), self.ctx)

    def children(self):
        """Return a list containing a single element self.body()

        >>> f = Function('f', IntSort(), IntSort())
        >>> x = Int('x')
        >>> q = ForAll(x, f(x) == 0)
        >>> q.children()
        [f(Var(0)) == 0]
        """
        return [self.body()]