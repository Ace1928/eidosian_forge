import os, sys, subprocess
import dis
import itertools
import numpy as np
import numba
from numba import jit, njit
from numba.core import errors, ir, types, typing, typeinfer, utils
from numba.core.typeconv import Conversion
from numba.extending import overload_method
from numba.tests.support import TestCase, tag
from numba.tests.test_typeconv import CompatibilityTestMixin
from numba.core.untyped_passes import TranslateByteCode, IRProcessing
from numba.core.typed_passes import PartialTypeInference
from numba.core.compiler_machinery import FunctionPass, register_pass
import unittest
@staticmethod
def _actually_test_complex_unify():

    def pyfunc(a):
        res = 0.0
        for i in range(len(a)):
            res += a[i]
        return res
    argtys = (types.Array(c128, 1, 'C'),)
    cfunc = njit(argtys)(pyfunc)
    return (pyfunc, cfunc)