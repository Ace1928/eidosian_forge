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
def _replace_multi_dim_ind(self, ind_var, index_set, new_index):
    """
        replace individual indices in multi-dimensional access variable, which
        is a build_tuple
        """
    pass_states = self.pass_states
    require(ind_var is not None)
    require(isinstance(pass_states.typemap[ind_var.name], (types.Tuple, types.UniTuple)))
    ind_def_node = get_definition(pass_states.func_ir, ind_var)
    require(isinstance(ind_def_node, ir.Expr) and ind_def_node.op == 'build_tuple')
    ind_def_node.items = [new_index if v.name in index_set else v for v in ind_def_node.items]