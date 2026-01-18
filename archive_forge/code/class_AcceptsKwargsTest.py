import functools
from oslotest import base as test_base
from oslo_utils import reflection
class AcceptsKwargsTest(test_base.BaseTestCase):

    def test_no_kwargs(self):
        self.assertEqual(False, reflection.accepts_kwargs(mere_function))

    def test_with_kwargs(self):
        self.assertEqual(True, reflection.accepts_kwargs(function_with_kwargs))