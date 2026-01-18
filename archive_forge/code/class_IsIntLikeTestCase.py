import collections.abc
import copy
import math
from unittest import mock
import ddt
from oslotest import base as test_base
import testscenarios
from oslo_utils import strutils
from oslo_utils import units
class IsIntLikeTestCase(test_base.BaseTestCase):

    def test_is_int_like_true(self):
        self.assertTrue(strutils.is_int_like(1))
        self.assertTrue(strutils.is_int_like('1'))
        self.assertTrue(strutils.is_int_like('514'))
        self.assertTrue(strutils.is_int_like('0'))

    def test_is_int_like_false(self):
        self.assertFalse(strutils.is_int_like(1.1))
        self.assertFalse(strutils.is_int_like('1.1'))
        self.assertFalse(strutils.is_int_like('1.1.1'))
        self.assertFalse(strutils.is_int_like(None))
        self.assertFalse(strutils.is_int_like('0.'))
        self.assertFalse(strutils.is_int_like('aaaaaa'))
        self.assertFalse(strutils.is_int_like('....'))
        self.assertFalse(strutils.is_int_like('1g'))
        self.assertFalse(strutils.is_int_like('0cc3346e-9fef-4445-abe6-5d2b2690ec64'))
        self.assertFalse(strutils.is_int_like('a1'))
        self.assertFalse(strutils.is_int_like('12e3'))
        self.assertFalse(strutils.is_int_like('0o51'))
        self.assertFalse(strutils.is_int_like('0xDEADBEEF'))