import inspect
import keyword
import unittest
from collections import namedtuple
from unittest import mock
from bpython import autocomplete, inspection
from bpython.line import LinePart
class TestMagicMethodCompletion(unittest.TestCase):

    def test_magic_methods_complete_after_double_underscores(self):
        com = autocomplete.MagicMethodCompletion()
        block = 'class Something(object)\n    def __'
        self.assertSetEqual(com.matches(10, '    def __', current_block=block, complete_magic_methods=True), set(autocomplete.MAGIC_METHODS))