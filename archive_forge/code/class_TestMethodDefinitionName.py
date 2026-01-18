import re
from typing import Optional, Tuple
import unittest
from bpython.line import (
class TestMethodDefinitionName(LineTestCase):

    def setUp(self):
        self.func = current_method_definition_name

    def test_simple(self):
        self.assertAccess('def <foo|>')
        self.assertAccess('    def bar(x, y)|:')
        self.assertAccess('    def <bar|>(x, y)')