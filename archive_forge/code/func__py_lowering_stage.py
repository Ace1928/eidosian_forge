from numba.core import (types, typing, funcdesc, config, pylowering, transforms,
from numba.core.compiler_machinery import (FunctionPass, LoweringPass,
from collections import defaultdict
import warnings
def _py_lowering_stage(self, targetctx, library, interp, flags):
    fndesc = funcdesc.PythonFunctionDescriptor.from_object_mode_function(interp)
    with targetctx.push_code_library(library):
        lower = pylowering.PyLower(targetctx, library, fndesc, interp)
        lower.lower()
        if not flags.no_cpython_wrapper:
            lower.create_cpython_wrapper()
        env = lower.env
        call_helper = lower.call_helper
        del lower
    from numba.core.compiler import _LowerResult
    if flags.no_compile:
        return _LowerResult(fndesc, call_helper, cfunc=None, env=env)
    else:
        cfunc = targetctx.get_executable(library, fndesc, env)
        return _LowerResult(fndesc, call_helper, cfunc=cfunc, env=env)