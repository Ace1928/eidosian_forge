import itertools
import numpy as np
import sys
from collections import namedtuple
from io import StringIO
from numba import njit, typeof, prange
from numba.core import (
from numba.tests.support import (TestCase, tag, skip_parfors_unsupported,
from numba.parfors.array_analysis import EquivSet, ArrayAnalysis
from numba.core.compiler import Compiler, Flags, PassManager
from numba.core.ir_utils import remove_dead
from numba.core.untyped_passes import (ExtractByteCode, TranslateByteCode, FixupArgs,
from numba.core.typed_passes import (NopythonTypeInference, AnnotateTypes,
from numba.core.compiler_machinery import FunctionPass, PassManager, register_pass
from numba.experimental import jitclass
import unittest
def _has_shapecall(self, func_ir, x):
    for label, block in func_ir.blocks.items():
        for expr in block.find_exprs(op='getattr'):
            if expr.attr == 'shape':
                y = func_ir.get_definition(expr.value, lhs_only=True)
                z = func_ir.get_definition(x, lhs_only=True)
                y = y.name if isinstance(y, ir.Var) else y
                z = z.name if isinstance(z, ir.Var) else z
                if y == z:
                    return True
    return False