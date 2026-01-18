from numba.tests.support import TestCase, linux_only
from numba.tests.gdb_support import needs_gdb, skip_unless_pexpect, GdbMIDriver
from unittest.mock import patch, Mock
from numba.core import datamodel
import numpy as np
from numba import typeof
import ctypes as ct
import unittest
class DIDerivedType_tuple:

    def __init__(self, the_tuple):
        self._type = DW_TAG_array_type(0, len(the_tuple) - 1)
        self._tuple = the_tuple

    @property
    def type(self):
        return self._type

    def __getitem__(self, item):
        return self._tuple[item]