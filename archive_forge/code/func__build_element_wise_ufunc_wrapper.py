import inspect
import warnings
from contextlib import contextmanager
from numba.core import config, targetconfig
from numba.core.decorators import jit
from numba.core.descriptors import TargetDescriptor
from numba.core.extending import is_jitted
from numba.core.errors import NumbaDeprecationWarning
from numba.core.options import TargetOptions, include_default_options
from numba.core.registry import cpu_target
from numba.core.target_extension import dispatcher_registry, target_registry
from numba.core import utils, types, serialize, compiler, sigutils
from numba.np.numpy_support import as_dtype
from numba.np.ufunc import _internal
from numba.np.ufunc.sigparse import parse_signature
from numba.np.ufunc.wrappers import build_ufunc_wrapper, build_gufunc_wrapper
from numba.core.caching import FunctionCache, NullCache
from numba.core.compiler_lock import global_compiler_lock
def _build_element_wise_ufunc_wrapper(cres, signature):
    """Build a wrapper for the ufunc loop entry point given by the
    compilation result object, using the element-wise signature.
    """
    ctx = cres.target_context
    library = cres.library
    fname = cres.fndesc.llvm_func_name
    with global_compiler_lock:
        info = build_ufunc_wrapper(library, ctx, fname, signature, cres.objectmode, cres)
        ptr = info.library.get_pointer_to_function(info.name)
    dtypenums = [as_dtype(a).num for a in signature.args]
    dtypenums.append(as_dtype(signature.return_type).num)
    return (dtypenums, ptr, cres.environment)