import sys
import subprocess
import numpy as np
import os
import warnings
from numba import jit, njit, types
from numba.core import errors
from numba.experimental import structref
from numba.extending import (overload, intrinsic, overload_method,
from numba.core.compiler import CompilerBase
from numba.core.untyped_passes import (TranslateByteCode, FixupArgs,
from numba.core.typed_passes import (NopythonTypeInference, DeadCodeElimination,
from numba.core.compiler_machinery import PassManager
from numba.core.types.functions import _err_reasons as error_reasons
from numba.tests.support import (skip_parfors_unsupported, override_config,
import unittest
def _run_in_separate_process(self, runcode, env):
    code = f'if 1:\n            {runcode}\n\n            '
    proc_env = os.environ.copy()
    proc_env.update(env)
    popen = subprocess.Popen([sys.executable, '-Wall', '-c', code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=proc_env)
    out, err = popen.communicate()
    if popen.returncode != 0:
        raise AssertionError('process failed with code %s: stderr follows\n%s\n' % (popen.returncode, err.decode()))
    return (out, err)