import pytest
from tests_python.debugger_unittest import IS_PY36_OR_GREATER, IS_CPYTHON
from tests_python.debug_constants import TEST_CYTHON
import opcode
import unittest
from _pydevd_frame_eval.vendored.bytecode import (
from _pydevd_frame_eval.vendored.bytecode.tests import TestCase
class SetLinenoTests(TestCase):

    def test_lineno(self):
        lineno = SetLineno(1)
        self.assertEqual(lineno.lineno, 1)

    def test_equality(self):
        lineno = SetLineno(1)
        self.assertNotEqual(lineno, 1)
        self.assertEqual(lineno, SetLineno(1))
        self.assertNotEqual(lineno, SetLineno(2))