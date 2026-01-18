import pickle
import sys
from oslo_utils import encodeutils
from taskflow import exceptions
from taskflow import test
from taskflow.tests import utils as test_utils
from taskflow.types import failure
class ReCreatedFailureTestCase(test.TestCase, GeneralFailureObjTestsMixin):

    def setUp(self):
        super(ReCreatedFailureTestCase, self).setUp()
        fail_obj = _captured_failure('Woot!')
        self.fail_obj = failure.Failure(exception_str=fail_obj.exception_str, traceback_str=fail_obj.traceback_str, exc_type_names=list(fail_obj))

    def test_value_lost(self):
        self.assertIsNone(self.fail_obj.exception)

    def test_no_exc_info(self):
        self.assertIsNone(self.fail_obj.exc_info)

    def test_pformat_traceback(self):
        text = self.fail_obj.pformat(traceback=True)
        self.assertIn('Traceback (most recent call last):', text)

    def test_reraises(self):
        exc = self.assertRaises(exceptions.WrappedFailure, self.fail_obj.reraise)
        self.assertIs(exc.check(RuntimeError), RuntimeError)

    def test_no_type_names(self):
        fail_obj = _captured_failure('Woot!')
        fail_obj = failure.Failure(exception_str=fail_obj.exception_str, traceback_str=fail_obj.traceback_str, exc_type_names=[])
        self.assertEqual([], list(fail_obj))
        self.assertEqual('Failure: Woot!', fail_obj.pformat())