import math
import numpy as np
from numba import int32, uint32, float32, float64, jit, vectorize
from numba.tests.support import tag, CheckWarningsMixin
import unittest
class TestCPUVectorizeJitted(unittest.TestCase, BaseVectorizeDecor):
    target = 'cpu'
    wrapper = jit(nopython=True)