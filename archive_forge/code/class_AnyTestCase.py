import builtins
import functools
import io
import time
from unittest import mock
import ddt
from os_brick import exception
from os_brick.tests import base
from os_brick import utils
@ddt.ddt
class AnyTestCase(base.TestCase):

    @ddt.data('hola', 1, None, {'a': 1}, {1, 2}, False)
    def test_equal(self, what):
        self.assertEqual(what, utils.ANY)
        self.assertEqual(utils.ANY, what)

    @ddt.data('hola', 1, None, {'a': 1}, {1, 2}, False)
    def test_different(self, what):
        self.assertFalse(what != utils.ANY)
        self.assertFalse(utils.ANY != what)
        self.assertFalse(utils.ANY > what)
        self.assertFalse(utils.ANY < what)
        self.assertFalse(utils.ANY <= what)
        self.assertFalse(utils.ANY >= what)