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
class NoPythonSupportedFeatureValidation(AnalysisPass):
    """NoPython Mode check: Validates the IR to ensure that features in use are
    in a form that is supported"""
    _name = 'nopython_supported_feature_validation'

    def __init__(self):
        AnalysisPass.__init__(self)

    def run_pass(self, state):
        raise_on_unsupported_feature(state.func_ir, state.typemap)
        warn_deprecated(state.func_ir, state.typemap)
        return False