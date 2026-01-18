from numba.tests import test_runtests
from numba import njit
def _check_cpu_compilation():

    @njit
    def foo(x):
        return x + 1
    result = foo(1)
    if result != 2:
        msg = f'Unexpected result from trial compilation. Expected: 2, Got: {result}.'
        raise AssertionError(msg)