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
class _CompileStatus(object):
    """
    Describes the state of compilation. Used like a C record.
    """
    __slots__ = ['fail_reason', 'can_fallback']

    def __init__(self, can_fallback):
        self.fail_reason = None
        self.can_fallback = can_fallback

    def __repr__(self):
        vals = []
        for k in self.__slots__:
            vals.append('{k}={v}'.format(k=k, v=getattr(self, k)))
        return ', '.join(vals)