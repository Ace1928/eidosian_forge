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
class ParforPass(ParforPassStates):
    """ParforPass class is responsible for converting NumPy
    calls in Numba intermediate representation to Parfors, which
    will lower into either sequential or parallel loops during lowering
    stage.
    """

    def _pre_run(self):
        self.array_analysis.run(self.func_ir.blocks)
        ir_utils._the_max_label.update(ir_utils.find_max_label(self.func_ir.blocks))

    def run(self):
        """run parfor conversion pass: replace Numpy calls
        with Parfors when possible and optimize the IR."""
        self._pre_run()
        if self.options.stencil:
            stencil_pass = StencilPass(self.func_ir, self.typemap, self.calltypes, self.array_analysis, self.typingctx, self.targetctx, self.flags)
            stencil_pass.run()
        if self.options.setitem:
            ConvertSetItemPass(self).run(self.func_ir.blocks)
        if self.options.numpy:
            ConvertNumpyPass(self).run(self.func_ir.blocks)
        if self.options.reduction:
            ConvertReducePass(self).run(self.func_ir.blocks)
        if self.options.prange:
            ConvertLoopPass(self).run(self.func_ir.blocks)
        if self.options.inplace_binop:
            ConvertInplaceBinop(self).run(self.func_ir.blocks)
        self.diagnostics.setup(self.func_ir, self.options.fusion)
        dprint_func_ir(self.func_ir, 'after parfor pass')

    def _find_mask(self, arr_def):
        """check if an array is of B[...M...], where M is a
        boolean array, and other indices (if available) are ints.
        If found, return B, M, M's type, and a tuple representing mask indices.
        Otherwise, raise GuardException.
        """
        return _find_mask(self.typemap, self.func_ir, arr_def)

    def _mk_parfor_loops(self, size_vars, scope, loc):
        """
        Create loop index variables and build LoopNest objects for a parfor.
        """
        return _mk_parfor_loops(self.typemap, size_vars, scope, loc)