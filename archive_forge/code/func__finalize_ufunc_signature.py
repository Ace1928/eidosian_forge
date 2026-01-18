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
def _finalize_ufunc_signature(cres, args, return_type):
    """Given a compilation result, argument types, and a return type,
    build a valid Numba signature after validating that it doesn't
    violate the constraints for the compilation mode.
    """
    if return_type is None:
        if cres.objectmode:
            raise TypeError('return type must be specified for object mode')
        else:
            return_type = cres.signature.return_type
    assert return_type != types.pyobject
    return return_type(*args)