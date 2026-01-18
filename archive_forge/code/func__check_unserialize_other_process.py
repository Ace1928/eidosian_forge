import warnings
import base64
import ctypes
import pickle
import re
import subprocess
import sys
import weakref
import llvmlite.binding as ll
import unittest
from numba import njit
from numba.core.codegen import JITCPUCodegen
from numba.core.compiler_lock import global_compiler_lock
from numba.tests.support import TestCase
def _check_unserialize_other_process(self, state):
    arg = base64.b64encode(pickle.dumps(state, -1))
    code = 'if 1:\n            import base64\n            import pickle\n            import sys\n            from numba.tests.test_codegen import %(test_class)s\n\n            state = pickle.loads(base64.b64decode(sys.argv[1]))\n            %(test_class)s._check_unserialize_sum(state)\n            ' % dict(test_class=self.__class__.__name__)
    subprocess.check_call([sys.executable, '-c', code, arg.decode()])