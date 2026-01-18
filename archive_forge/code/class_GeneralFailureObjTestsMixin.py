import pickle
import sys
from oslo_utils import encodeutils
from taskflow import exceptions
from taskflow import test
from taskflow.tests import utils as test_utils
from taskflow.types import failure
class GeneralFailureObjTestsMixin(object):

    def test_captures_message(self):
        self.assertEqual('Woot!', self.fail_obj.exception_str)

    def test_str(self):
        self.assertEqual('Failure: RuntimeError: Woot!', str(self.fail_obj))

    def test_exception_types(self):
        self.assertEqual(test_utils.RUNTIME_ERROR_CLASSES[:-2], list(self.fail_obj))

    def test_pformat_no_traceback(self):
        text = self.fail_obj.pformat()
        self.assertNotIn('Traceback', text)

    def test_check_str(self):
        val = 'Exception'
        self.assertEqual(val, self.fail_obj.check(val))

    def test_check_str_not_there(self):
        val = 'ValueError'
        self.assertIsNone(self.fail_obj.check(val))

    def test_check_type(self):
        self.assertIs(self.fail_obj.check(RuntimeError), RuntimeError)

    def test_check_type_not_there(self):
        self.assertIsNone(self.fail_obj.check(ValueError))