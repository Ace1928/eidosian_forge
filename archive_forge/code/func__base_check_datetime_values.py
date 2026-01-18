import sys
from itertools import product
import numpy as np
import unittest
from numba.core import types
from numba.core.errors import NumbaNotImplementedError
from numba.tests.support import TestCase
from numba.tests.enum_usecases import Shake, RequestError
from numba.np import numpy_support
def _base_check_datetime_values(self, func, np_type, nb_type):
    f = func
    for unit in ['', 'Y', 'M', 'D', 'h', 'm', 's', 'ms', 'us', 'ns', 'ps', 'fs', 'as']:
        if unit:
            t = np_type(3, unit)
        else:
            t = np_type('Nat')
        tp = f(t)
        self.assertEqual(tp, nb_type(unit))