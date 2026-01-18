from numba.tests.support import TestCase, linux_only
from numba.tests.gdb_support import needs_gdb, skip_unless_pexpect, GdbMIDriver
from unittest.mock import patch, Mock
from numba.core import datamodel
import numpy as np
from numba import typeof
import ctypes as ct
import unittest
class SelectedInferior:

    def read_memory(self, data, extent):
        buf = (ct.c_char * extent).from_address(data)
        return buf.raw