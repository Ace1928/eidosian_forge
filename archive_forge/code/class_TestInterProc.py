import gc
from numba import jit, int32
import unittest
class TestInterProc(unittest.TestCase):

    def test_bar_call_foo(self):
        global cfoo
        cfoo = jit((int32, int32), nopython=True)(foo)
        cbar = jit((int32, int32), nopython=True)(bar)
        self.assertEqual(cbar(1, 2), 1 + 2 + 2)

    def test_bar_call_foo_compiled_twice(self):
        global cfoo
        for i in range(2):
            cfoo = jit((int32, int32), nopython=True)(foo)
            gc.collect()
        cbar = jit((int32, int32), nopython=True)(bar)
        self.assertEqual(cbar(1, 2), 1 + 2 + 2)

    def test_callsite_compilation(self):
        self.assertEqual(outer(1, 2), 1 + 2)