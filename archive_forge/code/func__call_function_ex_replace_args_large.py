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
def _call_function_ex_replace_args_large(old_body, vararg_stmt, new_body, search_end, func_ir, errmsg, already_deleted_defs):
    """
    Extracts the args passed as vararg
    for CALL_FUNCTION_EX. This pass is taken when
    n_args > 30 and the bytecode looks like:

        BUILD_TUPLE # Create a list to append to
        # Start for each argument
        LOAD_FAST  # Load each argument.
        LIST_APPEND # Add the argument to the list
        # End for each argument
        ...
        LIST_TO_TUPLE # Convert the args to a tuple.

    In the IR generated, the tuple is created by concatenating
    together several 1 element tuples to an initial empty tuple.
    We traverse backwards in the IR, collecting args, until we
    find the original empty tuple. For example, the IR might
    look like:

        $orig_tuple = build_tuple(items=[])
        $first_var = build_tuple(items=[Var(arg0, test.py:6)])
        $next_tuple = $orig_tuple + $first_var
        ...
        $final_var = build_tuple(items=[Var(argn, test.py:6)])
        $final_tuple = $prev_tuple + $final_var
        $varargs_var = $final_tuple
    """
    search_start = 0
    total_args = []
    if isinstance(vararg_stmt, ir.Assign) and isinstance(vararg_stmt.value, ir.Var):
        target_name = vararg_stmt.value.name
        new_body[search_end] = None
        _remove_assignment_definition(old_body, search_end, func_ir, already_deleted_defs)
        search_end -= 1
    else:
        raise AssertionError('unreachable')
    while search_end >= search_start:
        concat_stmt = old_body[search_end]
        if isinstance(concat_stmt, ir.Assign) and concat_stmt.target.name == target_name and isinstance(concat_stmt.value, ir.Expr) and (concat_stmt.value.op == 'build_tuple') and (not concat_stmt.value.items):
            new_body[search_end] = None
            _remove_assignment_definition(old_body, search_end, func_ir, already_deleted_defs)
            break
        else:
            if search_end == search_start or not (isinstance(concat_stmt, ir.Assign) and concat_stmt.target.name == target_name and isinstance(concat_stmt.value, ir.Expr) and (concat_stmt.value.op == 'binop') and (concat_stmt.value.fn == operator.add)):
                raise UnsupportedError(errmsg)
            lhs_name = concat_stmt.value.lhs.name
            rhs_name = concat_stmt.value.rhs.name
            arg_tuple_stmt = old_body[search_end - 1]
            if not (isinstance(arg_tuple_stmt, ir.Assign) and isinstance(arg_tuple_stmt.value, ir.Expr) and (arg_tuple_stmt.value.op == 'build_tuple') and (len(arg_tuple_stmt.value.items) == 1)):
                raise UnsupportedError(errmsg)
            if arg_tuple_stmt.target.name == lhs_name:
                raise AssertionError('unreachable')
            elif arg_tuple_stmt.target.name == rhs_name:
                target_name = lhs_name
            else:
                raise UnsupportedError(errmsg)
            total_args.append(arg_tuple_stmt.value.items[0])
            new_body[search_end] = None
            new_body[search_end - 1] = None
            _remove_assignment_definition(old_body, search_end, func_ir, already_deleted_defs)
            _remove_assignment_definition(old_body, search_end - 1, func_ir, already_deleted_defs)
            search_end -= 2
            keep_looking = True
            while search_end >= search_start and keep_looking:
                next_stmt = old_body[search_end]
                if isinstance(next_stmt, ir.Assign) and next_stmt.target.name == target_name:
                    keep_looking = False
                else:
                    search_end -= 1
    if search_end == search_start:
        raise UnsupportedError(errmsg)
    return total_args[::-1]