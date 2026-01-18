import pickle
import sys
from oslo_utils import encodeutils
from taskflow import exceptions
from taskflow import test
from taskflow.tests import utils as test_utils
from taskflow.types import failure
class FailureCausesTest(test.TestCase):

    @classmethod
    def _raise_many(cls, messages):
        if not messages:
            return
        msg = messages.pop(0)
        e = RuntimeError(msg)
        try:
            cls._raise_many(messages)
            raise e
        except RuntimeError as e1:
            raise e from e1

    def test_causes(self):
        f = None
        try:
            self._raise_many(['Still still not working', 'Still not working', 'Not working'])
        except RuntimeError:
            f = failure.Failure()
        self.assertIsNotNone(f)
        self.assertEqual(2, len(f.causes))
        self.assertEqual('Still not working', f.causes[0].exception_str)
        self.assertEqual('Not working', f.causes[1].exception_str)
        f = f.causes[0]
        self.assertEqual(1, len(f.causes))
        self.assertEqual('Not working', f.causes[0].exception_str)
        f = f.causes[0]
        self.assertEqual(0, len(f.causes))

    def test_causes_to_from_dict(self):
        f = None
        try:
            self._raise_many(['Still still not working', 'Still not working', 'Not working'])
        except RuntimeError:
            f = failure.Failure()
        self.assertIsNotNone(f)
        d_f = f.to_dict()
        failure.Failure.validate(d_f)
        f = failure.Failure.from_dict(d_f)
        self.assertEqual(2, len(f.causes))
        self.assertEqual('Still not working', f.causes[0].exception_str)
        self.assertEqual('Not working', f.causes[1].exception_str)
        f = f.causes[0]
        self.assertEqual(1, len(f.causes))
        self.assertEqual('Not working', f.causes[0].exception_str)
        f = f.causes[0]
        self.assertEqual(0, len(f.causes))

    def test_causes_pickle(self):
        f = None
        try:
            self._raise_many(['Still still not working', 'Still not working', 'Not working'])
        except RuntimeError:
            f = failure.Failure()
        self.assertIsNotNone(f)
        p_f = pickle.dumps(f)
        f = pickle.loads(p_f)
        self.assertEqual(2, len(f.causes))
        self.assertEqual('Still not working', f.causes[0].exception_str)
        self.assertEqual('Not working', f.causes[1].exception_str)
        f = f.causes[0]
        self.assertEqual(1, len(f.causes))
        self.assertEqual('Not working', f.causes[0].exception_str)
        f = f.causes[0]
        self.assertEqual(0, len(f.causes))

    def test_causes_suppress_context(self):
        f = None
        try:
            try:
                self._raise_many(['Still still not working', 'Still not working', 'Not working'])
            except RuntimeError as e:
                raise e from None
        except RuntimeError:
            f = failure.Failure()
        self.assertIsNotNone(f)
        self.assertEqual([], list(f.causes))