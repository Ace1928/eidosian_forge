import numpy as np
import numba
from numba.tests.support import TestCase
class TestDynFunc(TestCase):

    def test_issue_455(self):
        inst = Issue455()
        inst.create_f()
        a = inst.call_f()
        self.assertPreciseEqual(a, np.ones_like(a))