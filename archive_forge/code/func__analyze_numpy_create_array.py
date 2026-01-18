import numpy
import operator
from numba.core import types, ir, config, cgutils, errors
from numba.core.ir_utils import (
from numba.core.analysis import compute_cfg_from_blocks
from numba.core.typing import npydecl, signature
import copy
from numba.core.extending import intrinsic
import llvmlite
def _analyze_numpy_create_array(self, scope, equiv_set, loc, args, kws):
    shape_var = None
    if len(args) > 0:
        shape_var = args[0]
    elif 'shape' in kws:
        shape_var = kws['shape']
    if shape_var:
        return ArrayAnalysis.AnalyzeResult(shape=shape_var)
    raise errors.UnsupportedRewriteError('Must specify a shape for array creation', loc=loc)