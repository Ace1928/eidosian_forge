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
def _remove_assignment_definition(old_body, idx, func_ir, already_deleted_defs):
    """
    Deletes the definition defined for old_body at index idx
    from func_ir. We assume this stmt will be deleted from
    new_body.

    In some optimizations we may update the same variable multiple times.
    In this situation, we only need to delete a particular definition once,
    this is tracked in already_deleted_def, which is a map from
    assignment name to the set of values that have already been
    deleted.
    """
    lhs = old_body[idx].target.name
    rhs = old_body[idx].value
    if rhs in func_ir._definitions[lhs]:
        func_ir._definitions[lhs].remove(rhs)
        already_deleted_defs[lhs].add(rhs)
    elif rhs not in already_deleted_defs[lhs]:
        raise UnsupportedError('Inconsistency found in the definitions while executing a peephole optimization. This suggests an internal error or inconsistency elsewhere in the compiler.')