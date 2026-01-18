import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
import opcode
import unittest
from _pydevd_frame_eval.vendored.bytecode import (
from _pydevd_frame_eval.vendored.bytecode.tests import TestCase
class VariableTests(TestCase):

    def test_str(self):
        for cls in (CellVar, FreeVar):
            var = cls('a')
            self.assertEqual(str(var), 'a')

    def test_repr(self):
        for cls in (CellVar, FreeVar):
            var = cls('_a_x_a_')
            r = repr(var)
            self.assertIn('_a_x_a_', r)
            self.assertIn(cls.__name__, r)

    def test_eq(self):
        f1 = FreeVar('a')
        f2 = FreeVar('b')
        c1 = CellVar('a')
        c2 = CellVar('b')
        for v1, v2, eq in ((f1, f1, True), (f1, f2, False), (f1, c1, False), (c1, c1, True), (c1, c2, False)):
            if eq:
                self.assertEqual(v1, v2)
            else:
                self.assertNotEqual(v1, v2)