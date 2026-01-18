import builtins
import collections
import dis
import operator
import logging
import textwrap
from numba.core import errors, ir, config
from numba.core.errors import NotDefinedError, UnsupportedError, error_extras
from numba.core.ir_utils import get_definition, guard
from numba.core.utils import (PYVERSION, BINOPS_TO_OPERATORS,
from numba.core.byteflow import Flow, AdaptDFA, AdaptCFA, BlockKind
from numba.core.unsafe import eh
from numba.cpython.unsafe.tuple import unpack_single_tuple
def _build_new_build_map(func_ir, name, old_body, old_lineno, new_items):
    """
    Create a new build_map with a new set of key/value items
    but all the other info the same.
    """
    old_assign = old_body[old_lineno]
    old_target = old_assign.target
    old_bm = old_assign.value
    literal_keys = []
    values = []
    for pair in new_items:
        k, v = pair
        key_def = guard(get_definition, func_ir, k)
        if isinstance(key_def, (ir.Const, ir.Global, ir.FreeVar)):
            literal_keys.append(key_def.value)
        value_def = guard(get_definition, func_ir, v)
        if isinstance(value_def, (ir.Const, ir.Global, ir.FreeVar)):
            values.append(value_def.value)
        else:
            values.append(_UNKNOWN_VALUE(v.name))
    value_indexes = {}
    if len(literal_keys) == len(new_items):
        literal_value = {x: y for x, y in zip(literal_keys, values)}
        for i, k in enumerate(literal_keys):
            value_indexes[k] = i
    else:
        literal_value = None
    new_bm = ir.Expr.build_map(items=new_items, size=len(new_items), literal_value=literal_value, value_indexes=value_indexes, loc=old_bm.loc)
    func_ir._definitions[name].append(new_bm)
    return ir.Assign(new_bm, ir.Var(old_target.scope, name, old_target.loc), new_bm.loc)