import os
import platform
import re
import subprocess
import sys
import threading
from itertools import permutations
from numba import njit, gdb, gdb_init, gdb_breakpoint, prange
from numba.core import errors
from numba import jit
from numba.tests.support import (TestCase, captured_stdout, tag,
from numba.tests.gdb_support import needs_gdb
import unittest
@not_arm
@unix_only
@needs_gdb
class TestGdbMisc(TestCase):

    @long_running
    def test_call_gdb_twice(self):

        def gen(f1, f2):

            @njit
            def impl():
                a = 1
                f1()
                b = 2
                f2()
                return a + b
            return impl
        msg_head = 'Calling either numba.gdb() or numba.gdb_init() more than'

        def check(func):
            with self.assertRaises(errors.UnsupportedError) as raises:
                func()
            self.assertIn(msg_head, str(raises.exception))
        for g1, g2 in permutations([gdb, gdb_init]):
            func = gen(g1, g2)
            check(func)

        @njit
        def use_globals():
            a = 1
            gdb()
            b = 2
            gdb_init()
            return a + b
        check(use_globals)