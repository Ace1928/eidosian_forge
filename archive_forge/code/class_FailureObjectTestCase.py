import pickle
import sys
from oslo_utils import encodeutils
from taskflow import exceptions
from taskflow import test
from taskflow.tests import utils as test_utils
from taskflow.types import failure
class FailureObjectTestCase(test.TestCase):

    def test_invalids(self):
        f = {'exception_str': 'blah', 'traceback_str': 'blah', 'exc_type_names': []}
        self.assertRaises(exceptions.InvalidFormat, failure.Failure.validate, f)
        f = {'exception_str': 'blah', 'exc_type_names': ['Exception']}
        self.assertRaises(exceptions.InvalidFormat, failure.Failure.validate, f)
        f = {'exception_str': 'blah', 'traceback_str': 'blah', 'exc_type_names': ['Exception'], 'version': -1}
        self.assertRaises(exceptions.InvalidFormat, failure.Failure.validate, f)

    def test_valid_from_dict_to_dict(self):
        f = _captured_failure('Woot!')
        d_f = f.to_dict()
        failure.Failure.validate(d_f)
        f2 = failure.Failure.from_dict(d_f)
        self.assertTrue(f.matches(f2))

    def test_bad_root_exception(self):
        f = _captured_failure('Woot!')
        d_f = f.to_dict()
        d_f['exc_type_names'] = ['Junk']
        self.assertRaises(exceptions.InvalidFormat, failure.Failure.validate, d_f)

    def test_valid_from_dict_to_dict_2(self):
        f = _captured_failure('Woot!')
        d_f = f.to_dict()
        d_f['exc_type_names'] = ['RuntimeError', 'Exception', 'BaseException']
        failure.Failure.validate(d_f)

    def test_cause_exception_args(self):
        f = _captured_failure('Woot!')
        d_f = f.to_dict()
        self.assertEqual(1, len(d_f['exc_args']))
        self.assertEqual(('Woot!',), d_f['exc_args'])
        f2 = failure.Failure.from_dict(d_f)
        self.assertEqual(f.exception_args, f2.exception_args)

    def test_dont_catch_base_exception(self):
        try:
            raise SystemExit()
        except BaseException:
            self.assertRaises(TypeError, failure.Failure)

    def test_unknown_argument(self):
        exc = self.assertRaises(TypeError, failure.Failure, exception_str='Woot!', traceback_str=None, exc_type_names=['Exception'], hi='hi there')
        expected = 'Failure.__init__ got unexpected keyword argument(s): hi'
        self.assertEqual(expected, str(exc))

    def test_empty_does_not_reraise(self):
        self.assertIsNone(failure.Failure.reraise_if_any([]))

    def test_reraises_one(self):
        fls = [_captured_failure('Woot!')]
        self.assertRaisesRegex(RuntimeError, '^Woot!$', failure.Failure.reraise_if_any, fls)

    def test_reraises_several(self):
        fls = [_captured_failure('Woot!'), _captured_failure('Oh, not again!')]
        exc = self.assertRaises(exceptions.WrappedFailure, failure.Failure.reraise_if_any, fls)
        self.assertEqual(fls, list(exc))

    def test_failure_copy(self):
        fail_obj = _captured_failure('Woot!')
        copied = fail_obj.copy()
        self.assertIsNot(fail_obj, copied)
        self.assertEqual(fail_obj, copied)
        self.assertTrue(fail_obj.matches(copied))

    def test_failure_copy_recaptured(self):
        captured = _captured_failure('Woot!')
        fail_obj = failure.Failure(exception_str=captured.exception_str, traceback_str=captured.traceback_str, exc_type_names=list(captured))
        copied = fail_obj.copy()
        self.assertIsNot(fail_obj, copied)
        self.assertEqual(fail_obj, copied)
        self.assertFalse(fail_obj != copied)
        self.assertTrue(fail_obj.matches(copied))

    def test_recaptured_not_eq(self):
        captured = _captured_failure('Woot!')
        fail_obj = failure.Failure(exception_str=captured.exception_str, traceback_str=captured.traceback_str, exc_type_names=list(captured), exc_args=list(captured.exception_args))
        self.assertFalse(fail_obj == captured)
        self.assertTrue(fail_obj != captured)
        self.assertTrue(fail_obj.matches(captured))

    def test_two_captured_eq(self):
        captured = _captured_failure('Woot!')
        captured2 = _captured_failure('Woot!')
        self.assertEqual(captured, captured2)

    def test_two_recaptured_neq(self):
        captured = _captured_failure('Woot!')
        fail_obj = failure.Failure(exception_str=captured.exception_str, traceback_str=captured.traceback_str, exc_type_names=list(captured))
        new_exc_str = captured.exception_str.replace('Woot', 'w00t')
        fail_obj2 = failure.Failure(exception_str=new_exc_str, traceback_str=captured.traceback_str, exc_type_names=list(captured))
        self.assertNotEqual(fail_obj, fail_obj2)
        self.assertFalse(fail_obj2.matches(fail_obj))

    def test_compares_to_none(self):
        captured = _captured_failure('Woot!')
        self.assertIsNotNone(captured)
        self.assertFalse(captured.matches(None))

    def test_pformat_traceback(self):
        captured = _captured_failure('Woot!')
        text = captured.pformat(traceback=True)
        self.assertIn('Traceback (most recent call last):', text)

    def test_pformat_traceback_captured_no_exc_info(self):
        captured = _captured_failure('Woot!')
        captured = failure.Failure.from_dict(captured.to_dict())
        text = captured.pformat(traceback=True)
        self.assertIn('Traceback (most recent call last):', text)

    def test_no_capture_exc_args(self):
        captured = _captured_failure(Exception('I am not valid JSON'))
        fail_obj = failure.Failure(exception_str=captured.exception_str, traceback_str=captured.traceback_str, exc_type_names=list(captured), exc_args=list(captured.exception_args))
        fail_json = fail_obj.to_dict(include_args=False)
        self.assertNotEqual(fail_obj.exception_args, fail_json['exc_args'])
        self.assertEqual(fail_json['exc_args'], tuple())