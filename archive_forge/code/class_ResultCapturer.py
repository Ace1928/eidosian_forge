from collections import namedtuple
import numpy as np
from numba.tests.support import (TestCase, MemoryLeakMixin,
from numba import njit, typed, literal_unroll, prange
from numba.core import types, errors, ir
from numba.testing import unittest
from numba.core.extending import overload
from numba.core.compiler_machinery import (PassManager, register_pass,
from numba.core.compiler import CompilerBase
from numba.core.untyped_passes import (FixupArgs, TranslateByteCode,
from numba.core.typed_passes import (NopythonTypeInference, IRLegalization,
from numba.core.ir_utils import (compute_cfg_from_blocks, flatten_labels)
from numba.core.types.functions import _header_lead
@register_pass(mutates_CFG=False, analysis_only=True)
class ResultCapturer(AnalysisPass):
    _name = 'capture_%s' % real_pass._name
    _real_pass = real_pass

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        result = real_pass().run_pass(state)
        mutation_results = state.metadata.setdefault('mutation_results', {})
        mutation_results[real_pass] = result
        return result