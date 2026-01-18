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
@classmethod
def _check_unserialize_sum(cls, state):
    codegen = JITCPUCodegen('other_codegen')
    library = codegen.unserialize_library(state)
    ptr = library.get_pointer_to_function('sum')
    assert ptr, ptr
    cfunc = ctypes_sum_ty(ptr)
    res = cfunc(2, 3)
    assert res == 5, res