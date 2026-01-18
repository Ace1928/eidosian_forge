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
def _make_index_var(typemap, scope, index_vars, body_block, force_tuple=False):
    """ When generating a SetItem call to an array in a parfor, the general
    strategy is to generate a tuple if the array is more than 1 dimension.
    If it is 1 dimensional then you can use a simple variable.  This routine
    is also used when converting pndindex to parfor but pndindex requires a
    tuple even if the iteration space is 1 dimensional.  The pndindex use of
    this function will use force_tuple to make the output index a tuple even
    if it is one dimensional.
    """
    ndims = len(index_vars)
    loc = body_block.loc
    if ndims > 1 or force_tuple:
        tuple_var = ir.Var(scope, mk_unique_var('$parfor_index_tuple_var'), loc)
        typemap[tuple_var.name] = types.containers.UniTuple(types.uintp, ndims)
        tuple_call = ir.Expr.build_tuple(list(index_vars), loc)
        tuple_assign = ir.Assign(tuple_call, tuple_var, loc)
        body_block.body.append(tuple_assign)
        return (tuple_var, types.containers.UniTuple(types.uintp, ndims))
    elif ndims == 1:
        return (index_vars[0], types.uintp)
    else:
        raise errors.UnsupportedRewriteError('Parfor does not handle arrays of dimension 0', loc=loc)