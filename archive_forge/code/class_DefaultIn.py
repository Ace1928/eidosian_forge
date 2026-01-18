import numpy as np
from numba import cuda
from numba.cuda.args import wrap_arg
from numba.cuda.testing import CUDATestCase
import unittest
class DefaultIn(object):

    def prepare_args(self, ty, val, **kwargs):
        return (ty, wrap_arg(val, default=cuda.In))