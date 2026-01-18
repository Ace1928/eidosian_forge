from collections import namedtuple
import copy
import warnings
from numba.core.tracing import event
from numba.core import (utils, errors, interpreter, bytecode, postproc, config,
from numba.parfors.parfor import ParforDiagnostics
from numba.core.errors import CompilerError
from numba.core.environment import lookup_environment
from numba.core.compiler_machinery import PassManager
from numba.core.untyped_passes import (ExtractByteCode, TranslateByteCode,
from numba.core.typed_passes import (NopythonTypeInference, AnnotateTypes,
from numba.core.object_mode_passes import (ObjectModeFrontEnd,
from numba.core.targetconfig import TargetConfig, Option, ConfigStack
class DefaultPassBuilder(object):
    """
    This is the default pass builder, it contains the "classic" default
    pipelines as pre-canned PassManager instances:
      - nopython
      - objectmode
      - interpreted
      - typed
      - untyped
      - nopython lowering
    """

    @staticmethod
    def define_nopython_pipeline(state, name='nopython'):
        """Returns an nopython mode pipeline based PassManager
        """
        dpb = DefaultPassBuilder
        pm = PassManager(name)
        untyped_passes = dpb.define_untyped_pipeline(state)
        pm.passes.extend(untyped_passes.passes)
        typed_passes = dpb.define_typed_pipeline(state)
        pm.passes.extend(typed_passes.passes)
        lowering_passes = dpb.define_nopython_lowering_pipeline(state)
        pm.passes.extend(lowering_passes.passes)
        pm.finalize()
        return pm

    @staticmethod
    def define_nopython_lowering_pipeline(state, name='nopython_lowering'):
        pm = PassManager(name)
        pm.add_pass(NoPythonSupportedFeatureValidation, 'ensure features that are in use are in a valid form')
        pm.add_pass(IRLegalization, 'ensure IR is legal prior to lowering')
        pm.add_pass(AnnotateTypes, 'annotate types')
        if state.flags.auto_parallel.enabled:
            pm.add_pass(NativeParforLowering, 'native parfor lowering')
        else:
            pm.add_pass(NativeLowering, 'native lowering')
        pm.add_pass(NoPythonBackend, 'nopython mode backend')
        pm.add_pass(DumpParforDiagnostics, 'dump parfor diagnostics')
        pm.finalize()
        return pm

    @staticmethod
    def define_parfor_gufunc_nopython_lowering_pipeline(state, name='parfor_gufunc_nopython_lowering'):
        pm = PassManager(name)
        pm.add_pass(NoPythonSupportedFeatureValidation, 'ensure features that are in use are in a valid form')
        pm.add_pass(IRLegalization, 'ensure IR is legal prior to lowering')
        pm.add_pass(AnnotateTypes, 'annotate types')
        if state.flags.auto_parallel.enabled:
            pm.add_pass(NativeParforLowering, 'native parfor lowering')
        else:
            pm.add_pass(NativeLowering, 'native lowering')
        pm.add_pass(NoPythonBackend, 'nopython mode backend')
        pm.finalize()
        return pm

    @staticmethod
    def define_typed_pipeline(state, name='typed'):
        """Returns the typed part of the nopython pipeline"""
        pm = PassManager(name)
        pm.add_pass(NopythonTypeInference, 'nopython frontend')
        pm.add_pass(PreLowerStripPhis, 'remove phis nodes')
        pm.add_pass(InlineOverloads, 'inline overloaded functions')
        if state.flags.auto_parallel.enabled:
            pm.add_pass(PreParforPass, 'Preprocessing for parfors')
        if not state.flags.no_rewrites:
            pm.add_pass(NopythonRewrites, 'nopython rewrites')
        if state.flags.auto_parallel.enabled:
            pm.add_pass(ParforPass, 'convert to parfors')
            pm.add_pass(ParforFusionPass, 'fuse parfors')
            pm.add_pass(ParforPreLoweringPass, 'parfor prelowering')
        pm.finalize()
        return pm

    @staticmethod
    def define_parfor_gufunc_pipeline(state, name='parfor_gufunc_typed'):
        """Returns the typed part of the nopython pipeline"""
        pm = PassManager(name)
        assert state.func_ir
        pm.add_pass(IRProcessing, 'processing IR')
        pm.add_pass(NopythonTypeInference, 'nopython frontend')
        pm.add_pass(ParforPreLoweringPass, 'parfor prelowering')
        pm.finalize()
        return pm

    @staticmethod
    def define_untyped_pipeline(state, name='untyped'):
        """Returns an untyped part of the nopython pipeline"""
        pm = PassManager(name)
        if config.USE_RVSDG_FRONTEND:
            if state.func_ir is None:
                pm.add_pass(RVSDGFrontend, 'rvsdg frontend')
                pm.add_pass(FixupArgs, 'fix up args')
            pm.add_pass(IRProcessing, 'processing IR')
        else:
            if state.func_ir is None:
                pm.add_pass(TranslateByteCode, 'analyzing bytecode')
                pm.add_pass(FixupArgs, 'fix up args')
            pm.add_pass(IRProcessing, 'processing IR')
        pm.add_pass(WithLifting, 'Handle with contexts')
        pm.add_pass(InlineClosureLikes, 'inline calls to locally defined closures')
        if not state.flags.no_rewrites:
            pm.add_pass(RewriteSemanticConstants, 'rewrite semantic constants')
            pm.add_pass(DeadBranchPrune, 'dead branch pruning')
            pm.add_pass(GenericRewrites, 'nopython rewrites')
        pm.add_pass(RewriteDynamicRaises, 'rewrite dynamic raises')
        pm.add_pass(MakeFunctionToJitFunction, 'convert make_function into JIT functions')
        pm.add_pass(InlineInlinables, 'inline inlinable functions')
        if not state.flags.no_rewrites:
            pm.add_pass(DeadBranchPrune, 'dead branch pruning')
        pm.add_pass(FindLiterallyCalls, 'find literally calls')
        pm.add_pass(LiteralUnroll, 'handles literal_unroll')
        if state.flags.enable_ssa:
            pm.add_pass(ReconstructSSA, 'ssa')
        pm.add_pass(LiteralPropagationSubPipelinePass, 'Literal propagation')
        pm.finalize()
        return pm

    @staticmethod
    def define_objectmode_pipeline(state, name='object'):
        """Returns an object-mode pipeline based PassManager
        """
        pm = PassManager(name)
        if state.func_ir is None:
            pm.add_pass(TranslateByteCode, 'analyzing bytecode')
            pm.add_pass(FixupArgs, 'fix up args')
        else:
            pm.add_pass(PreLowerStripPhis, 'remove phis nodes')
        pm.add_pass(IRProcessing, 'processing IR')
        pm.add_pass(CanonicalizeLoopEntry, 'canonicalize loop entry')
        pm.add_pass(CanonicalizeLoopExit, 'canonicalize loop exit')
        pm.add_pass(ObjectModeFrontEnd, 'object mode frontend')
        pm.add_pass(InlineClosureLikes, 'inline calls to locally defined closures')
        pm.add_pass(MakeFunctionToJitFunction, 'convert make_function into JIT functions')
        pm.add_pass(IRLegalization, 'ensure IR is legal prior to lowering')
        pm.add_pass(AnnotateTypes, 'annotate types')
        pm.add_pass(ObjectModeBackEnd, 'object mode backend')
        pm.finalize()
        return pm