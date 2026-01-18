from numba.tests.support import TestCase, linux_only
from numba.tests.gdb_support import needs_gdb, skip_unless_pexpect, GdbMIDriver
from unittest.mock import patch, Mock
from numba.core import datamodel
import numpy as np
from numba import typeof
import ctypes as ct
import unittest
def _subprocess_test_runner(self, test_mod):
    themod = f'numba.tests.gdb.{test_mod}'
    self.subprocess_test_runner(test_module=themod, test_class='Test', test_name='test', envvars=self._NUMBA_OPT_0_ENV)