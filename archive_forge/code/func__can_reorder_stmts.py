import types as pytypes  # avoid confusion with numba.types
import sys, math
import os
import textwrap
import copy
import inspect
import linecache
from functools import reduce
from collections import defaultdict, OrderedDict, namedtuple
from contextlib import contextmanager
import operator
from dataclasses import make_dataclass
import warnings
from llvmlite import ir as lir
from numba.core.imputils import impl_ret_untracked
import numba.core.ir
from numba.core import types, typing, utils, errors, ir, analysis, postproc, rewrites, typeinfer, config, ir_utils
from numba import prange, pndindex
from numba.np.npdatetime_helpers import datetime_minimum, datetime_maximum
from numba.np.numpy_support import as_dtype, numpy_version
from numba.core.typing.templates import infer_global, AbstractTemplate
from numba.stencils.stencilparfor import StencilPass
from numba.core.extending import register_jitable, lower_builtin
from numba.core.ir_utils import (
from numba.core.analysis import (compute_use_defs, compute_live_map,
from numba.core.controlflow import CFGraph
from numba.core.typing import npydecl, signature
from numba.core.types.functions import Function
from numba.parfors.array_analysis import (random_int_args, random_1arg_size,
from numba.core.extending import overload
import copy
import numpy
import numpy as np
from numba.parfors import array_analysis
import numba.cpython.builtins
from numba.stencils import stencilparfor
def _can_reorder_stmts(stmt, next_stmt, func_ir, call_table, alias_map, arg_aliases):
    """
    Check dependencies to determine if a parfor can be reordered in the IR block
    with a non-parfor statement.
    """
    if isinstance(stmt, Parfor) and (not isinstance(next_stmt, Parfor)) and (not isinstance(next_stmt, ir.Print)) and (not isinstance(next_stmt, ir.Assign) or has_no_side_effect(next_stmt.value, set(), call_table) or guard(is_assert_equiv, func_ir, next_stmt.value)):
        stmt_accesses = expand_aliases({v.name for v in stmt.list_vars()}, alias_map, arg_aliases)
        stmt_writes = expand_aliases(get_parfor_writes(stmt), alias_map, arg_aliases)
        next_accesses = expand_aliases({v.name for v in next_stmt.list_vars()}, alias_map, arg_aliases)
        next_writes = expand_aliases(get_stmt_writes(next_stmt), alias_map, arg_aliases)
        if len(stmt_writes & next_accesses | next_writes & stmt_accesses) == 0:
            return True
    return False