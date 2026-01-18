import sys
from numba import cfunc, jit
from numba.tests.support import TestCase, captured_stderr

    Tests for functionality of this module's cfuncs.
    Note this does not define any "test_*" method, instead check_module()
    should be called by hand.
    