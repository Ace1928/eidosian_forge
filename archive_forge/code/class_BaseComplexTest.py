import math
import itertools
import numpy as np
from numba.cuda.testing import unittest, CUDATestCase
from numba.core import types
from numba import cuda
from numba.tests.complex_usecases import (real_usecase, imag_usecase,
from numba.np import numpy_support
class BaseComplexTest(CUDATestCase):

    def basic_values(self):
        reals = [-0.0, +0.0, 1, -1, +1.5, -3.5, float('-inf'), float('+inf'), float('nan')]
        return [complex(x, y) for x, y in itertools.product(reals, reals)]

    def more_values(self):
        reals = [0.0, +0.0, 1, -1, -math.pi, +math.pi, float('-inf'), float('+inf'), float('nan')]
        return [complex(x, y) for x, y in itertools.product(reals, reals)]

    def non_nan_values(self):
        reals = [-0.0, +0.0, 1, -1, -math.pi, +math.pi, float('inf'), float('-inf')]
        return [complex(x, y) for x, y in itertools.product(reals, reals)]

    def run_func(self, pyfunc, sigs, values, ulps=1, ignore_sign_on_zero=False):
        for sig in sigs:
            if isinstance(sig, types.Type):
                sig = (sig,)
            if isinstance(sig, tuple):
                sig = sig[0](*sig)
            prec = 'single' if sig.args[0] in (types.float32, types.complex64) else 'double'
            cudafunc = compile_scalar_func(pyfunc, sig.args, sig.return_type)
            ok_values = []
            expected_list = []
            for args in values:
                if not isinstance(args, (list, tuple)):
                    args = (args,)
                try:
                    expected_list.append(pyfunc(*args))
                    ok_values.append(args)
                except ValueError as e:
                    self.assertIn('math domain error', str(e))
                    continue
            got_list = cudafunc(ok_values)
            for got, expected, args in zip(got_list, expected_list, ok_values):
                msg = 'for input %r with prec %r' % (args, prec)
                self.assertPreciseEqual(got, expected, prec=prec, ulps=ulps, ignore_sign_on_zero=ignore_sign_on_zero, msg=msg)
    run_unary = run_func
    run_binary = run_func