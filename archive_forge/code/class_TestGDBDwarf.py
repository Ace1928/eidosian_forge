from numba.tests.support import TestCase, linux_only
from numba.tests.gdb_support import needs_gdb, skip_unless_pexpect, GdbMIDriver
from unittest.mock import patch, Mock
from numba.core import datamodel
import numpy as np
from numba import typeof
import ctypes as ct
import unittest
@linux_only
@needs_gdb
@skip_unless_pexpect
class TestGDBDwarf(TestCase):
    _NUMBA_OPT_0_ENV = {'NUMBA_OPT': '0'}

    def _gdb_has_numpy(self):
        """Returns True if gdb has NumPy support, False otherwise"""
        driver = GdbMIDriver(__file__, debug=False)
        has_numpy = driver.supports_numpy()
        driver.quit()
        return has_numpy

    def _subprocess_test_runner(self, test_mod):
        themod = f'numba.tests.gdb.{test_mod}'
        self.subprocess_test_runner(test_module=themod, test_class='Test', test_name='test', envvars=self._NUMBA_OPT_0_ENV)

    def test_basic(self):
        self._subprocess_test_runner('test_basic')

    def test_array_arg(self):
        self._subprocess_test_runner('test_array_arg')

    def test_conditional_breakpoint(self):
        self._subprocess_test_runner('test_conditional_breakpoint')

    def test_break_on_symbol(self):
        self._subprocess_test_runner('test_break_on_symbol')

    def test_break_on_symbol_version(self):
        self._subprocess_test_runner('test_break_on_symbol_version')

    def test_pretty_print(self):
        if not self._gdb_has_numpy():
            _msg = 'Cannot find gdb with NumPy support'
            self.skipTest(_msg)
        self._subprocess_test_runner('test_pretty_print')