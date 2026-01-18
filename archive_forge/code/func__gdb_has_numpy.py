from numba.tests.support import TestCase, linux_only
from numba.tests.gdb_support import needs_gdb, skip_unless_pexpect, GdbMIDriver
from unittest.mock import patch, Mock
from numba.core import datamodel
import numpy as np
from numba import typeof
import ctypes as ct
import unittest
def _gdb_has_numpy(self):
    """Returns True if gdb has NumPy support, False otherwise"""
    driver = GdbMIDriver(__file__, debug=False)
    has_numpy = driver.supports_numpy()
    driver.quit()
    return has_numpy