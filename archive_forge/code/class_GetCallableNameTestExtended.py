import functools
from oslotest import base as test_base
from oslo_utils import reflection
class GetCallableNameTestExtended(test_base.BaseTestCase):

    class InnerCallableClass(object):

        def __call__(self):
            pass

    def test_inner_callable_class(self):
        obj = self.InnerCallableClass()
        name = reflection.get_callable_name(obj.__call__)
        expected_name = '.'.join((__name__, 'GetCallableNameTestExtended', 'InnerCallableClass', '__call__'))
        self.assertEqual(expected_name, name)

    def test_inner_callable_function(self):

        def a():

            def b():
                pass
            return b
        name = reflection.get_callable_name(a())
        expected_name = '.'.join((__name__, 'GetCallableNameTestExtended', 'test_inner_callable_function', '<locals>', 'a', '<locals>', 'b'))
        self.assertEqual(expected_name, name)

    def test_inner_class(self):
        obj = self.InnerCallableClass()
        name = reflection.get_callable_name(obj)
        expected_name = '.'.join((__name__, 'GetCallableNameTestExtended', 'InnerCallableClass'))
        self.assertEqual(expected_name, name)