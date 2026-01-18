import functools
from oslotest import base as test_base
from oslo_utils import reflection
class BoundMethodTest(test_base.BaseTestCase):

    def test_baddy(self):
        b = BadClass()
        self.assertTrue(reflection.is_bound_method(b.do_something))

    def test_static_method(self):
        self.assertFalse(reflection.is_bound_method(Class.static_method))