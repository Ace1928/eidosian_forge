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
class ParforPassStates:
    """This class encapsulates all internal states of the ParforPass.
    """

    def __init__(self, func_ir, typemap, calltypes, return_type, typingctx, targetctx, options, flags, metadata, diagnostics=ParforDiagnostics()):
        self.func_ir = func_ir
        self.typemap = typemap
        self.calltypes = calltypes
        self.typingctx = typingctx
        self.targetctx = targetctx
        self.return_type = return_type
        self.options = options
        self.diagnostics = diagnostics
        self.swapped_fns = diagnostics.replaced_fns
        self.fusion_info = diagnostics.fusion_info
        self.nested_fusion_info = diagnostics.nested_fusion_info
        self.array_analysis = array_analysis.ArrayAnalysis(self.typingctx, self.func_ir, self.typemap, self.calltypes)
        ir_utils._the_max_label.update(max(func_ir.blocks.keys()))
        self.flags = flags
        self.metadata = metadata
        if 'parfors' not in metadata:
            metadata['parfors'] = {}