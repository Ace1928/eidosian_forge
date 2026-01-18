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
def _update_parfor_get_setitems(block_body, index_var, alias_map, saved_values, lives):
    """
    replace getitems of a previously set array in a block of parfor loop body
    """
    for stmt in block_body:
        if isinstance(stmt, (ir.StaticSetItem, ir.SetItem)) and get_index_var(stmt).name == index_var.name and (stmt.target.name not in lives):
            for w in alias_map.get(stmt.target.name, []):
                saved_values.pop(w, None)
            saved_values[stmt.target.name] = stmt.value
            continue
        if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr):
            rhs = stmt.value
            if rhs.op == 'getitem' and isinstance(rhs.index, ir.Var):
                if rhs.index.name == index_var.name:
                    stmt.value = saved_values.get(rhs.value.name, rhs)
                    continue
        for v in stmt.list_vars():
            saved_values.pop(v.name, None)
            for w in alias_map.get(v.name, []):
                saved_values.pop(w, None)
    return