import sys
import os
import multiprocessing as mp
import warnings
from numba.core.config import IS_WIN32, IS_OSX
from numba.core.errors import NumbaWarning
from numba.cuda.cudadrv import nvvm
from numba.cuda.testing import (
from numba.cuda.cuda_paths import (
def _fake_non_conda_env():
    """
    Monkeypatch sys.prefix to hide the fact we are in a conda-env
    """
    sys.prefix = ''