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
class AstMap:
    """A mapping from ASTs to ASTs."""

    def __init__(self, m=None, ctx=None):
        self.map = None
        if m is None:
            self.ctx = _get_ctx(ctx)
            self.map = Z3_mk_ast_map(self.ctx.ref())
        else:
            self.map = m
            assert ctx is not None
            self.ctx = ctx
        Z3_ast_map_inc_ref(self.ctx.ref(), self.map)

    def __deepcopy__(self, memo={}):
        return AstMap(self.map, self.ctx)

    def __del__(self):
        if self.map is not None and self.ctx.ref() is not None and (Z3_ast_map_dec_ref is not None):
            Z3_ast_map_dec_ref(self.ctx.ref(), self.map)

    def __len__(self):
        """Return the size of the map.

        >>> M = AstMap()
        >>> len(M)
        0
        >>> x = Int('x')
        >>> M[x] = IntVal(1)
        >>> len(M)
        1
        """
        return int(Z3_ast_map_size(self.ctx.ref(), self.map))

    def __contains__(self, key):
        """Return `True` if the map contains key `key`.

        >>> M = AstMap()
        >>> x = Int('x')
        >>> M[x] = x + 1
        >>> x in M
        True
        >>> x+1 in M
        False
        """
        return Z3_ast_map_contains(self.ctx.ref(), self.map, key.as_ast())

    def __getitem__(self, key):
        """Retrieve the value associated with key `key`.

        >>> M = AstMap()
        >>> x = Int('x')
        >>> M[x] = x + 1
        >>> M[x]
        x + 1
        """
        return _to_ast_ref(Z3_ast_map_find(self.ctx.ref(), self.map, key.as_ast()), self.ctx)

    def __setitem__(self, k, v):
        """Add/Update key `k` with value `v`.

        >>> M = AstMap()
        >>> x = Int('x')
        >>> M[x] = x + 1
        >>> len(M)
        1
        >>> M[x]
        x + 1
        >>> M[x] = IntVal(1)
        >>> M[x]
        1
        """
        Z3_ast_map_insert(self.ctx.ref(), self.map, k.as_ast(), v.as_ast())

    def __repr__(self):
        return Z3_ast_map_to_string(self.ctx.ref(), self.map)

    def erase(self, k):
        """Remove the entry associated with key `k`.

        >>> M = AstMap()
        >>> x = Int('x')
        >>> M[x] = x + 1
        >>> len(M)
        1
        >>> M.erase(x)
        >>> len(M)
        0
        """
        Z3_ast_map_erase(self.ctx.ref(), self.map, k.as_ast())

    def reset(self):
        """Remove all entries from the map.

        >>> M = AstMap()
        >>> x = Int('x')
        >>> M[x]   = x + 1
        >>> M[x+x] = IntVal(1)
        >>> len(M)
        2
        >>> M.reset()
        >>> len(M)
        0
        """
        Z3_ast_map_reset(self.ctx.ref(), self.map)

    def keys(self):
        """Return an AstVector containing all keys in the map.

        >>> M = AstMap()
        >>> x = Int('x')
        >>> M[x]   = x + 1
        >>> M[x+x] = IntVal(1)
        >>> M.keys()
        [x, x + x]
        """
        return AstVector(Z3_ast_map_keys(self.ctx.ref(), self.map), self.ctx)