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
class AstVector(Z3PPObject):
    """A collection (vector) of ASTs."""

    def __init__(self, v=None, ctx=None):
        self.vector = None
        if v is None:
            self.ctx = _get_ctx(ctx)
            self.vector = Z3_mk_ast_vector(self.ctx.ref())
        else:
            self.vector = v
            assert ctx is not None
            self.ctx = ctx
        Z3_ast_vector_inc_ref(self.ctx.ref(), self.vector)

    def __del__(self):
        if self.vector is not None and self.ctx.ref() is not None and (Z3_ast_vector_dec_ref is not None):
            Z3_ast_vector_dec_ref(self.ctx.ref(), self.vector)

    def __len__(self):
        """Return the size of the vector `self`.

        >>> A = AstVector()
        >>> len(A)
        0
        >>> A.push(Int('x'))
        >>> A.push(Int('x'))
        >>> len(A)
        2
        """
        return int(Z3_ast_vector_size(self.ctx.ref(), self.vector))

    def __getitem__(self, i):
        """Return the AST at position `i`.

        >>> A = AstVector()
        >>> A.push(Int('x') + 1)
        >>> A.push(Int('y'))
        >>> A[0]
        x + 1
        >>> A[1]
        y
        """
        if isinstance(i, int):
            if i < 0:
                i += self.__len__()
            if i >= self.__len__():
                raise IndexError
            return _to_ast_ref(Z3_ast_vector_get(self.ctx.ref(), self.vector, i), self.ctx)
        elif isinstance(i, slice):
            result = []
            for ii in range(*i.indices(self.__len__())):
                result.append(_to_ast_ref(Z3_ast_vector_get(self.ctx.ref(), self.vector, ii), self.ctx))
            return result

    def __setitem__(self, i, v):
        """Update AST at position `i`.

        >>> A = AstVector()
        >>> A.push(Int('x') + 1)
        >>> A.push(Int('y'))
        >>> A[0]
        x + 1
        >>> A[0] = Int('x')
        >>> A[0]
        x
        """
        if i >= self.__len__():
            raise IndexError
        Z3_ast_vector_set(self.ctx.ref(), self.vector, i, v.as_ast())

    def push(self, v):
        """Add `v` in the end of the vector.

        >>> A = AstVector()
        >>> len(A)
        0
        >>> A.push(Int('x'))
        >>> len(A)
        1
        """
        Z3_ast_vector_push(self.ctx.ref(), self.vector, v.as_ast())

    def resize(self, sz):
        """Resize the vector to `sz` elements.

        >>> A = AstVector()
        >>> A.resize(10)
        >>> len(A)
        10
        >>> for i in range(10): A[i] = Int('x')
        >>> A[5]
        x
        """
        Z3_ast_vector_resize(self.ctx.ref(), self.vector, sz)

    def __contains__(self, item):
        """Return `True` if the vector contains `item`.

        >>> x = Int('x')
        >>> A = AstVector()
        >>> x in A
        False
        >>> A.push(x)
        >>> x in A
        True
        >>> (x+1) in A
        False
        >>> A.push(x+1)
        >>> (x+1) in A
        True
        >>> A
        [x, x + 1]
        """
        for elem in self:
            if elem.eq(item):
                return True
        return False

    def translate(self, other_ctx):
        """Copy vector `self` to context `other_ctx`.

        >>> x = Int('x')
        >>> A = AstVector()
        >>> A.push(x)
        >>> c2 = Context()
        >>> B = A.translate(c2)
        >>> B
        [x]
        """
        return AstVector(Z3_ast_vector_translate(self.ctx.ref(), self.vector, other_ctx.ref()), ctx=other_ctx)

    def __copy__(self):
        return self.translate(self.ctx)

    def __deepcopy__(self, memo={}):
        return self.translate(self.ctx)

    def __repr__(self):
        return obj_to_string(self)

    def sexpr(self):
        """Return a textual representation of the s-expression representing the vector."""
        return Z3_ast_vector_to_string(self.ctx.ref(), self.vector)