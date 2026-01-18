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
def _make_subtarget(targetctx, flags):
    """
    Make a new target context from the given target context and flags.
    """
    subtargetoptions = {}
    if flags.debuginfo:
        subtargetoptions['enable_debuginfo'] = True
    if flags.boundscheck:
        subtargetoptions['enable_boundscheck'] = True
    if flags.nrt:
        subtargetoptions['enable_nrt'] = True
    if flags.auto_parallel:
        subtargetoptions['auto_parallel'] = flags.auto_parallel
    if flags.fastmath:
        subtargetoptions['fastmath'] = flags.fastmath
    error_model = callconv.create_error_model(flags.error_model, targetctx)
    subtargetoptions['error_model'] = error_model
    return targetctx.subtarget(**subtargetoptions)