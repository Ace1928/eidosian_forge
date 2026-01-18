from collections import OrderedDict
from functools import reduce, partial
from itertools import chain
from operator import attrgetter, mul
import math
import warnings
from ..units import (
from ..util.pyutil import deprecated
from ..util._expr import Expr, Symbol
from .rates import RateExpr, MassAction
def _reg_unique(expr, rxn=None):
    if not isinstance(expr, Expr):
        raise NotImplementedError('Currently only Expr sub classes are supported.')
    if isinstance(expr, MassAction):
        if expr.args is None:
            uk, = expr.unique_keys
            if uk not in substitutions:
                unique[uk] = None
                _reg_unique_unit(uk, _get_arg_dim(expr, rxn), 0)
                return
        else:
            arg, = expr.args
            if isinstance(arg, Symbol):
                uk, = arg.unique_keys
                if uk not in substitutions:
                    unique[uk] = None
                    _reg_unique_unit(uk, _get_arg_dim(expr, rxn), 0)
                    return
    if expr.args is None:
        for idx, k in enumerate(expr.unique_keys):
            if k not in substitutions:
                unique[k] = None
                _reg_unique_unit(k, _get_arg_dim(expr, rxn), idx)
    else:
        for idx, arg in enumerate(expr.args):
            if isinstance(arg, Expr):
                _reg_unique(arg, rxn=rxn)
            elif expr.unique_keys is not None and idx < len(expr.unique_keys):
                uk = expr.unique_keys[idx]
                if uk not in substitutions:
                    unique[uk] = arg
                    _reg_unique_unit(uk, _get_arg_dim(expr, rxn), idx)