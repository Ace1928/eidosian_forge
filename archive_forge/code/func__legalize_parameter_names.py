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
@contextlib.contextmanager
def _legalize_parameter_names(var_list):
    """
    Legalize names in the variable list for use as a Python function's
    parameter names.
    """
    var_map = OrderedDict()
    for var in var_list:
        old_name = var.name
        new_name = var.scope.redefine(old_name, loc=var.loc).name
        new_name = new_name.replace('$', '_').replace('.', '_')
        if new_name in var_map:
            raise AssertionError(f'{new_name!r} not unique')
        var_map[new_name] = (var, old_name)
        var.name = new_name
    param_names = list(var_map)
    try:
        yield param_names
    finally:
        for var, old_name in var_map.values():
            var.name = old_name