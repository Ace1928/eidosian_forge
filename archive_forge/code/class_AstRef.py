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
class AstRef(Z3PPObject):
    """AST are Direct Acyclic Graphs (DAGs) used to represent sorts, declarations and expressions."""

    def __init__(self, ast, ctx=None):
        self.ast = ast
        self.ctx = _get_ctx(ctx)
        Z3_inc_ref(self.ctx.ref(), self.as_ast())

    def __del__(self):
        if self.ctx.ref() is not None and self.ast is not None and (Z3_dec_ref is not None):
            Z3_dec_ref(self.ctx.ref(), self.as_ast())
            self.ast = None

    def __deepcopy__(self, memo={}):
        return _to_ast_ref(self.ast, self.ctx)

    def __str__(self):
        return obj_to_string(self)

    def __repr__(self):
        return obj_to_string(self)

    def __eq__(self, other):
        return self.eq(other)

    def __hash__(self):
        return self.hash()

    def __nonzero__(self):
        return self.__bool__()

    def __bool__(self):
        if is_true(self):
            return True
        elif is_false(self):
            return False
        elif is_eq(self) and self.num_args() == 2:
            return self.arg(0).eq(self.arg(1))
        else:
            raise Z3Exception('Symbolic expressions cannot be cast to concrete Boolean values.')

    def sexpr(self):
        """Return a string representing the AST node in s-expression notation.

        >>> x = Int('x')
        >>> ((x + 1)*x).sexpr()
        '(* (+ x 1) x)'
        """
        return Z3_ast_to_string(self.ctx_ref(), self.as_ast())

    def as_ast(self):
        """Return a pointer to the corresponding C Z3_ast object."""
        return self.ast

    def get_id(self):
        """Return unique identifier for object. It can be used for hash-tables and maps."""
        return Z3_get_ast_id(self.ctx_ref(), self.as_ast())

    def ctx_ref(self):
        """Return a reference to the C context where this AST node is stored."""
        return self.ctx.ref()

    def eq(self, other):
        """Return `True` if `self` and `other` are structurally identical.

        >>> x = Int('x')
        >>> n1 = x + 1
        >>> n2 = 1 + x
        >>> n1.eq(n2)
        False
        >>> n1 = simplify(n1)
        >>> n2 = simplify(n2)
        >>> n1.eq(n2)
        True
        """
        if z3_debug():
            _z3_assert(is_ast(other), 'Z3 AST expected')
        return Z3_is_eq_ast(self.ctx_ref(), self.as_ast(), other.as_ast())

    def translate(self, target):
        """Translate `self` to the context `target`. That is, return a copy of `self` in the context `target`.

        >>> c1 = Context()
        >>> c2 = Context()
        >>> x  = Int('x', c1)
        >>> y  = Int('y', c2)
        >>> # Nodes in different contexts can't be mixed.
        >>> # However, we can translate nodes from one context to another.
        >>> x.translate(c2) + y
        x + y
        """
        if z3_debug():
            _z3_assert(isinstance(target, Context), 'argument must be a Z3 context')
        return _to_ast_ref(Z3_translate(self.ctx.ref(), self.as_ast(), target.ref()), target)

    def __copy__(self):
        return self.translate(self.ctx)

    def hash(self):
        """Return a hashcode for the `self`.

        >>> n1 = simplify(Int('x') + 1)
        >>> n2 = simplify(2 + Int('x') - 1)
        >>> n1.hash() == n2.hash()
        True
        """
        return Z3_get_ast_hash(self.ctx_ref(), self.as_ast())