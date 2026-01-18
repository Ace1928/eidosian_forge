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
def _has_assertcall(self, func_ir, typemap, args):
    msg = 'Sizes of {} do not match'.format(', '.join(args))
    for label, block in func_ir.blocks.items():
        for expr in block.find_exprs(op='call'):
            fn = func_ir.get_definition(expr.func.name)
            if isinstance(fn, ir.Global) and fn.name == 'assert_equiv':
                typ = typemap[expr.args[0].name]
                if typ.literal_value.startswith(msg):
                    return True
    return False