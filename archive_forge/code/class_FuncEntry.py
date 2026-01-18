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
class FuncEntry:
    """Store the value of the interpretation of a function in a particular point."""

    def __init__(self, entry, ctx):
        self.entry = entry
        self.ctx = ctx
        Z3_func_entry_inc_ref(self.ctx.ref(), self.entry)

    def __deepcopy__(self, memo={}):
        return FuncEntry(self.entry, self.ctx)

    def __del__(self):
        if self.ctx.ref() is not None and Z3_func_entry_dec_ref is not None:
            Z3_func_entry_dec_ref(self.ctx.ref(), self.entry)

    def num_args(self):
        """Return the number of arguments in the given entry.

        >>> f = Function('f', IntSort(), IntSort(), IntSort())
        >>> s = Solver()
        >>> s.add(f(0, 1) == 10, f(1, 2) == 20, f(1, 0) == 10)
        >>> s.check()
        sat
        >>> m = s.model()
        >>> f_i = m[f]
        >>> f_i.num_entries()
        1
        >>> e = f_i.entry(0)
        >>> e.num_args()
        2
        """
        return int(Z3_func_entry_get_num_args(self.ctx.ref(), self.entry))

    def arg_value(self, idx):
        """Return the value of argument `idx`.

        >>> f = Function('f', IntSort(), IntSort(), IntSort())
        >>> s = Solver()
        >>> s.add(f(0, 1) == 10, f(1, 2) == 20, f(1, 0) == 10)
        >>> s.check()
        sat
        >>> m = s.model()
        >>> f_i = m[f]
        >>> f_i.num_entries()
        1
        >>> e = f_i.entry(0)
        >>> e
        [1, 2, 20]
        >>> e.num_args()
        2
        >>> e.arg_value(0)
        1
        >>> e.arg_value(1)
        2
        >>> try:
        ...   e.arg_value(2)
        ... except IndexError:
        ...   print("index error")
        index error
        """
        if idx >= self.num_args():
            raise IndexError
        return _to_expr_ref(Z3_func_entry_get_arg(self.ctx.ref(), self.entry, idx), self.ctx)

    def value(self):
        """Return the value of the function at point `self`.

        >>> f = Function('f', IntSort(), IntSort(), IntSort())
        >>> s = Solver()
        >>> s.add(f(0, 1) == 10, f(1, 2) == 20, f(1, 0) == 10)
        >>> s.check()
        sat
        >>> m = s.model()
        >>> f_i = m[f]
        >>> f_i.num_entries()
        1
        >>> e = f_i.entry(0)
        >>> e
        [1, 2, 20]
        >>> e.num_args()
        2
        >>> e.value()
        20
        """
        return _to_expr_ref(Z3_func_entry_get_value(self.ctx.ref(), self.entry), self.ctx)

    def as_list(self):
        """Return entry `self` as a Python list.
        >>> f = Function('f', IntSort(), IntSort(), IntSort())
        >>> s = Solver()
        >>> s.add(f(0, 1) == 10, f(1, 2) == 20, f(1, 0) == 10)
        >>> s.check()
        sat
        >>> m = s.model()
        >>> f_i = m[f]
        >>> f_i.num_entries()
        1
        >>> e = f_i.entry(0)
        >>> e.as_list()
        [1, 2, 20]
        """
        args = [self.arg_value(i) for i in range(self.num_args())]
        args.append(self.value())
        return args

    def __repr__(self):
        return repr(self.as_list())