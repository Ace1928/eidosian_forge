import unittest
from numba.tests.support import TestCase, run_in_subprocess
class TestNumbaImport(TestCase):
    """
    Test behaviour of importing Numba.
    """

    def test_laziness(self):
        """
        Importing top-level numba features should not import too many modules.
        """
        banlist = ['cffi', 'distutils', 'numba.cuda', 'numba.cpython.mathimpl', 'numba.cpython.randomimpl', 'numba.tests', 'numba.core.typing.collections', 'numba.core.typing.listdecl', 'numba.core.typing.npdatetime']
        for mod in banlist:
            if mod not in ('cffi',):
                __import__(mod)
        code = 'if 1:\n            from numba import jit, vectorize\n            from numba.core import types\n            import sys\n            print(list(sys.modules))\n            '
        out, _ = run_in_subprocess(code)
        modlist = set(eval(out.strip()))
        unexpected = set(banlist) & set(modlist)
        self.assertFalse(unexpected, 'some modules unexpectedly imported')

    def test_no_impl_import(self):
        """
        Tests that importing jit does not trigger import of modules containing
        lowering implementations that would likely install things in the
        builtins registry and have side effects impacting other targets
        """
        banlist = ['numba.cpython.slicing', 'numba.cpython.tupleobj', 'numba.cpython.enumimpl', 'numba.cpython.hashing', 'numba.cpython.heapq', 'numba.cpython.iterators', 'numba.cpython.numbers', 'numba.cpython.rangeobj', 'numba.cpython.cmathimpl', 'numba.cpython.mathimpl', 'numba.cpython.printimpl', 'numba.cpython.randomimpl', 'numba.core.optional', 'numba.misc.gdb_hook', 'numba.misc.literal', 'numba.misc.cffiimpl', 'numba.np.linalg', 'numba.np.polynomial', 'numba.np.arraymath', 'numba.np.npdatetime', 'numba.np.npyimpl', 'numba.typed.typeddict', 'numba.typed.typedlist', 'numba.experimental.jitclass.base']
        code1 = 'if 1:\n            import sys\n            import numba\n            print(list(sys.modules))\n            '
        code2 = 'if 1:\n            import sys\n            from numba import njit\n            @njit\n            def foo():\n                pass\n            print(list(sys.modules))\n            '
        for code in (code1, code2):
            out, _ = run_in_subprocess(code)
            modlist = set(eval(out.strip()))
            unexpected = set(banlist) & set(modlist)
            self.assertFalse(unexpected, 'some modules unexpectedly imported')

    def test_no_accidental_warnings(self):
        code = 'import numba'
        flags = ['-Werror', '-Wignore::DeprecationWarning:packaging.version:']
        run_in_subprocess(code, flags)

    def test_import_star(self):
        code = 'from numba import *'
        run_in_subprocess(code)