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
@register_pass(mutates_CFG=False, analysis_only=False)
class AnnotateTypes(AnalysisPass):
    _name = 'annotate_types'

    def __init__(self):
        AnalysisPass.__init__(self)

    def get_analysis_usage(self, AU):
        AU.add_required(IRLegalization)

    def run_pass(self, state):
        """
        Create type annotation after type inference
        """
        func_ir = state.func_ir.copy()
        state.type_annotation = type_annotations.TypeAnnotation(func_ir=func_ir, typemap=state.typemap, calltypes=state.calltypes, lifted=state.lifted, lifted_from=state.lifted_from, args=state.args, return_type=state.return_type, html_output=config.HTML)
        if config.ANNOTATE:
            print('ANNOTATION'.center(80, '-'))
            print(state.type_annotation)
            print('=' * 80)
        if config.HTML:
            with open(config.HTML, 'w') as fout:
                state.type_annotation.html_annotate(fout)
        return False