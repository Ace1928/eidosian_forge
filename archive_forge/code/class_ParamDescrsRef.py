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
class ParamDescrsRef:
    """Set of parameter descriptions for Solvers, Tactics and Simplifiers in Z3.
    """

    def __init__(self, descr, ctx=None):
        _z3_assert(isinstance(descr, ParamDescrs), 'parameter description object expected')
        self.ctx = _get_ctx(ctx)
        self.descr = descr
        Z3_param_descrs_inc_ref(self.ctx.ref(), self.descr)

    def __deepcopy__(self, memo={}):
        return ParamsDescrsRef(self.descr, self.ctx)

    def __del__(self):
        if self.ctx.ref() is not None and Z3_param_descrs_dec_ref is not None:
            Z3_param_descrs_dec_ref(self.ctx.ref(), self.descr)

    def size(self):
        """Return the size of in the parameter description `self`.
        """
        return int(Z3_param_descrs_size(self.ctx.ref(), self.descr))

    def __len__(self):
        """Return the size of in the parameter description `self`.
        """
        return self.size()

    def get_name(self, i):
        """Return the i-th parameter name in the parameter description `self`.
        """
        return _symbol2py(self.ctx, Z3_param_descrs_get_name(self.ctx.ref(), self.descr, i))

    def get_kind(self, n):
        """Return the kind of the parameter named `n`.
        """
        return Z3_param_descrs_get_kind(self.ctx.ref(), self.descr, to_symbol(n, self.ctx))

    def get_documentation(self, n):
        """Return the documentation string of the parameter named `n`.
        """
        return Z3_param_descrs_get_documentation(self.ctx.ref(), self.descr, to_symbol(n, self.ctx))

    def __getitem__(self, arg):
        if _is_int(arg):
            return self.get_name(arg)
        else:
            return self.get_kind(arg)

    def __repr__(self):
        return Z3_param_descrs_to_string(self.ctx.ref(), self.descr)