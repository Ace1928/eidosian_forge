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
def _add_liveness_return_block(blocks, lives, typemap):
    last_label = max(blocks.keys())
    return_label = last_label + 1
    loc = blocks[last_label].loc
    scope = blocks[last_label].scope
    blocks[return_label] = ir.Block(scope, loc)
    tuple_var = ir.Var(scope, mk_unique_var('$tuple_var'), loc)
    typemap[tuple_var.name] = types.containers.UniTuple(types.uintp, 2)
    live_vars = [ir.Var(scope, v, loc) for v in lives]
    tuple_call = ir.Expr.build_tuple(live_vars, loc)
    blocks[return_label].body.append(ir.Assign(tuple_call, tuple_var, loc))
    blocks[return_label].body.append(ir.Return(tuple_var, loc))
    return (return_label, tuple_var)