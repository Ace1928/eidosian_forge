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
def _ensure_critical_deps():
    """
    Make sure the Python, NumPy and SciPy present are supported versions.
    This has to be done _before_ importing anything from Numba such that
    incompatible versions can be reported to the user. If this occurs _after_
    importing things from Numba and there's an issue in e.g. a Numba c-ext, a
    SystemError might have occurred which prevents reporting the likely cause of
    the problem (incompatible versions of critical dependencies).
    """

    def extract_version(mod):
        return tuple(map(int, mod.__version__.split('.')[:2]))
    PYVERSION = sys.version_info[:2]
    if PYVERSION < (3, 9):
        msg = f'Numba needs Python 3.9 or greater. Got Python {PYVERSION[0]}.{PYVERSION[1]}.'
        raise ImportError(msg)
    import numpy as np
    numpy_version = extract_version(np)
    if numpy_version < (1, 22):
        msg = f'Numba needs NumPy 1.22 or greater. Got NumPy {numpy_version[0]}.{numpy_version[1]}.'
        raise ImportError(msg)
    elif numpy_version > (1, 26):
        raise ImportError('Numba needs NumPy 1.26 or less')
    try:
        import scipy
    except ImportError:
        pass
    else:
        sp_version = extract_version(scipy)
        if sp_version < (1, 0):
            msg = f'Numba requires SciPy version 1.0 or greater. Got SciPy {scipy.__version__}.'
            raise ImportError(msg)