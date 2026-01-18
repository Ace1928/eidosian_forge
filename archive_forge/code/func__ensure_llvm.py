import platform
import re
import sys
import warnings
from ._version import get_versions
from numba.misc.init_utils import generate_version_info
from numba.core import config
from numba.core import types, errors
from numba.misc.special import (
from numba.core.errors import *
import numba.core.types as types
from numba.core.types import *
from numba.core.decorators import (cfunc, jit, njit, stencil,
from numba.np.ufunc import (vectorize, guvectorize, threading_layer,
from numba.np.numpy_support import carray, farray, from_dtype
from numba import experimental
import numba.core.withcontexts
from numba.core.withcontexts import objmode_context as objmode
from numba.core.withcontexts import parallel_chunksize
import numba.core.target_extension
import numba.typed
import llvmlite
def _ensure_llvm():
    """
    Make sure llvmlite is operational.
    """
    import warnings
    import llvmlite
    regex = re.compile('(\\d+)\\.(\\d+).(\\d+)')
    m = regex.match(llvmlite.__version__)
    if m:
        ver = tuple(map(int, m.groups()))
        if ver < _min_llvmlite_version:
            msg = 'Numba requires at least version %d.%d.%d of llvmlite.\nInstalled version is %s.\nPlease update llvmlite.' % (_min_llvmlite_version + (llvmlite.__version__,))
            raise ImportError(msg)
    else:
        warnings.warn('llvmlite version format not recognized!')
    from llvmlite.binding import llvm_version_info, check_jit_execution
    if llvm_version_info < _min_llvm_version:
        msg = 'Numba requires at least version %d.%d.%d of LLVM.\nInstalled llvmlite is built against version %d.%d.%d.\nPlease update llvmlite.' % (_min_llvm_version + llvm_version_info)
        raise ImportError(msg)
    check_jit_execution()