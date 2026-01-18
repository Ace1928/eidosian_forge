import ast
from collections import defaultdict, OrderedDict
import contextlib
import sys
from types import SimpleNamespace
import numpy as np
import operator
from numba.core import types, targetconfig, ir, rewrites, compiler
from numba.core.typing import npydecl
from numba.np.ufunc.dufunc import DUFunc
class ExprKernel(npyimpl._Kernel):

    def generate(self, *args):
        arg_zip = zip(args, self.outer_sig.args, inner_sig.args)
        cast_args = [self.cast(val, inty, outty) for val, inty, outty in arg_zip]
        result = self.context.call_internal(builder, cres.fndesc, inner_sig, cast_args)
        return self.cast(result, inner_sig.return_type, self.outer_sig.return_type)