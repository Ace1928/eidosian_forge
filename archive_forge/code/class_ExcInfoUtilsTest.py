import pickle
import sys
from oslo_utils import encodeutils
from taskflow import exceptions
from taskflow import test
from taskflow.tests import utils as test_utils
from taskflow.types import failure
class ExcInfoUtilsTest(test.TestCase):

    def test_copy_none(self):
        result = failure._copy_exc_info(None)
        self.assertIsNone(result)

    def test_copy_exc_info(self):
        exc_info = _make_exc_info('Woot!')
        result = failure._copy_exc_info(exc_info)
        self.assertIsNot(result, exc_info)
        self.assertIs(result[0], RuntimeError)
        self.assertIsNot(result[1], exc_info[1])
        self.assertIs(result[2], exc_info[2])

    def test_none_equals(self):
        self.assertTrue(failure._are_equal_exc_info_tuples(None, None))

    def test_none_ne_tuple(self):
        exc_info = _make_exc_info('Woot!')
        self.assertFalse(failure._are_equal_exc_info_tuples(None, exc_info))

    def test_tuple_nen_none(self):
        exc_info = _make_exc_info('Woot!')
        self.assertFalse(failure._are_equal_exc_info_tuples(exc_info, None))

    def test_tuple_equals_itself(self):
        exc_info = _make_exc_info('Woot!')
        self.assertTrue(failure._are_equal_exc_info_tuples(exc_info, exc_info))

    def test_typle_equals_copy(self):
        exc_info = _make_exc_info('Woot!')
        copied = failure._copy_exc_info(exc_info)
        self.assertTrue(failure._are_equal_exc_info_tuples(exc_info, copied))