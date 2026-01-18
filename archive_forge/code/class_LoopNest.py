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
class LoopNest(object):
    """The LoopNest class holds information of a single loop including
    the index variable (of a non-negative integer value), and the
    range variable, e.g. range(r) is 0 to r-1 with step size 1.
    """

    def __init__(self, index_variable, start, stop, step):
        self.index_variable = index_variable
        self.start = start
        self.stop = stop
        self.step = step

    def __repr__(self):
        return 'LoopNest(index_variable = {}, range = ({}, {}, {}))'.format(self.index_variable, self.start, self.stop, self.step)

    def list_vars(self):
        all_uses = []
        all_uses.append(self.index_variable)
        if isinstance(self.start, ir.Var):
            all_uses.append(self.start)
        if isinstance(self.stop, ir.Var):
            all_uses.append(self.stop)
        if isinstance(self.step, ir.Var):
            all_uses.append(self.step)
        return all_uses