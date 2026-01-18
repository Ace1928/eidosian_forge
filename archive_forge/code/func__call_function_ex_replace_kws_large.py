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
def _call_function_ex_replace_kws_large(old_body, buildmap_name, buildmap_idx, search_end, new_body, func_ir, errmsg, already_deleted_defs):
    """
    Extracts the kws args passed as varkwarg
    for CALL_FUNCTION_EX. This pass is taken when
    n_kws > 15 and the bytecode looks like:

        BUILD_MAP # Construct the map
        # Start for each argument
        LOAD_CONST # Load a constant for the name of the argument
        LOAD_FAST  # Load each argument.
        MAP_ADD # Append the (key, value) pair to the map
        # End for each argument

    In the IR generated, the initial build map is empty and a series
    of setitems are applied afterwards. THE IR looks like:

        $build_map_var = build_map(items=[])
        $constvar = const(str, ...) # create the const key
        # CREATE THE ARGUMENT, This may take multiple lines.
        $created_arg = ...
        $var = getattr(
            value=$build_map_var,
            attr=__setitem__,
        )
        $unused_var = call $var($constvar, $created_arg)

    We iterate through the IR, deleting all usages of the buildmap
    from the new_body, and adds the kws to a new kws list.
    """
    new_body[buildmap_idx] = None
    _remove_assignment_definition(old_body, buildmap_idx, func_ir, already_deleted_defs)
    kws = []
    search_start = buildmap_idx + 1
    while search_start <= search_end:
        const_stmt = old_body[search_start]
        if not (isinstance(const_stmt, ir.Assign) and isinstance(const_stmt.value, ir.Const)):
            raise UnsupportedError(errmsg)
        key_var_name = const_stmt.target.name
        key_val = const_stmt.value.value
        search_start += 1
        found_getattr = False
        while search_start <= search_end and (not found_getattr):
            getattr_stmt = old_body[search_start]
            if isinstance(getattr_stmt, ir.Assign) and isinstance(getattr_stmt.value, ir.Expr) and (getattr_stmt.value.op == 'getattr') and (getattr_stmt.value.value.name == buildmap_name) and (getattr_stmt.value.attr == '__setitem__'):
                found_getattr = True
            else:
                search_start += 1
        if not found_getattr or search_start == search_end:
            raise UnsupportedError(errmsg)
        setitem_stmt = old_body[search_start + 1]
        if not (isinstance(setitem_stmt, ir.Assign) and isinstance(setitem_stmt.value, ir.Expr) and (setitem_stmt.value.op == 'call') and (setitem_stmt.value.func.name == getattr_stmt.target.name) and (len(setitem_stmt.value.args) == 2) and (setitem_stmt.value.args[0].name == key_var_name)):
            raise UnsupportedError(errmsg)
        arg_var = setitem_stmt.value.args[1]
        kws.append((key_val, arg_var))
        new_body[search_start] = None
        new_body[search_start + 1] = None
        _remove_assignment_definition(old_body, search_start, func_ir, already_deleted_defs)
        _remove_assignment_definition(old_body, search_start + 1, func_ir, already_deleted_defs)
        search_start += 2
    return kws