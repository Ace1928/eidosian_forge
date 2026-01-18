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
def _compile_and_test(self, fn, arg_tys, asserts=[], equivs=[], idempotent=True):
    """
        Compile the given function and get its IR.
        """
    test_pipeline = ArrayAnalysisTester.mk_pipeline(arg_tys)
    test_idempotence = self.compare_ir if idempotent else lambda x: ()
    analysis = test_pipeline.compile_to_ir(fn, test_idempotence)
    if equivs:
        for func in equivs:
            func(analysis.equiv_sets[0])
    if asserts is None:
        self.assertTrue(self._has_no_assertcall(analysis.func_ir))
    else:
        for func in asserts:
            func(analysis.func_ir, analysis.typemap)