import re
from typing import Optional, Tuple
import unittest
from bpython.line import (
class TestSingleWord(LineTestCase):

    def setUp(self):
        self.func = current_single_word

    def test_simple(self):
        self.assertAccess('foo.bar|')
        self.assertAccess('.foo|')
        self.assertAccess(' <foo|>')