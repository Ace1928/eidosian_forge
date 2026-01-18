from numba.tests import test_runtests
from numba import njit
def _check_runtests():
    test_inst = test_runtests.TestCase()
    test_inst.test_default()