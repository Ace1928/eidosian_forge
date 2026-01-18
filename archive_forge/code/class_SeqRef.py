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
class SeqRef(ExprRef):
    """Sequence expression."""

    def sort(self):
        return SeqSortRef(Z3_get_sort(self.ctx_ref(), self.as_ast()), self.ctx)

    def __add__(self, other):
        return Concat(self, other)

    def __radd__(self, other):
        return Concat(other, self)

    def __getitem__(self, i):
        if _is_int(i):
            i = IntVal(i, self.ctx)
        return _to_expr_ref(Z3_mk_seq_nth(self.ctx_ref(), self.as_ast(), i.as_ast()), self.ctx)

    def at(self, i):
        if _is_int(i):
            i = IntVal(i, self.ctx)
        return SeqRef(Z3_mk_seq_at(self.ctx_ref(), self.as_ast(), i.as_ast()), self.ctx)

    def is_string(self):
        return Z3_is_string_sort(self.ctx_ref(), Z3_get_sort(self.ctx_ref(), self.as_ast()))

    def is_string_value(self):
        return Z3_is_string(self.ctx_ref(), self.as_ast())

    def as_string(self):
        """Return a string representation of sequence expression."""
        if self.is_string_value():
            string_length = ctypes.c_uint()
            chars = Z3_get_lstring(self.ctx_ref(), self.as_ast(), byref(string_length))
            return string_at(chars, size=string_length.value).decode('latin-1')
        return Z3_ast_to_string(self.ctx_ref(), self.as_ast())

    def __le__(self, other):
        return _to_expr_ref(Z3_mk_str_le(self.ctx_ref(), self.as_ast(), other.as_ast()), self.ctx)

    def __lt__(self, other):
        return _to_expr_ref(Z3_mk_str_lt(self.ctx_ref(), self.as_ast(), other.as_ast()), self.ctx)

    def __ge__(self, other):
        return _to_expr_ref(Z3_mk_str_le(self.ctx_ref(), other.as_ast(), self.as_ast()), self.ctx)

    def __gt__(self, other):
        return _to_expr_ref(Z3_mk_str_lt(self.ctx_ref(), other.as_ast(), self.as_ast()), self.ctx)