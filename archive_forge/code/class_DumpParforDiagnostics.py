import abc
from contextlib import contextmanager
from collections import defaultdict, namedtuple
from functools import partial
from copy import copy
import warnings
from numba.core import (errors, types, typing, ir, funcdesc, rewrites,
from numba.parfors.parfor import PreParforPass as _parfor_PreParforPass
from numba.parfors.parfor import ParforPass as _parfor_ParforPass
from numba.parfors.parfor import ParforFusionPass as _parfor_ParforFusionPass
from numba.parfors.parfor import ParforPreLoweringPass as \
from numba.parfors.parfor import Parfor
from numba.parfors.parfor_lowering import ParforLower
from numba.core.compiler_machinery import (FunctionPass, LoweringPass,
from numba.core.annotations import type_annotations
from numba.core.ir_utils import (raise_on_unsupported_feature, warn_deprecated,
from numba.core import postproc
from llvmlite import binding as llvm
@register_pass(mutates_CFG=False, analysis_only=True)
class DumpParforDiagnostics(AnalysisPass):
    _name = 'dump_parfor_diagnostics'

    def __init__(self):
        AnalysisPass.__init__(self)

    def run_pass(self, state):
        if state.flags.auto_parallel.enabled:
            if config.PARALLEL_DIAGNOSTICS:
                if state.parfor_diagnostics is not None:
                    state.parfor_diagnostics.dump(config.PARALLEL_DIAGNOSTICS)
                else:
                    raise RuntimeError('Diagnostics failed.')
        return True