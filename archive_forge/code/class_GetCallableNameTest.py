import functools
from oslotest import base as test_base
from oslo_utils import reflection
class GetCallableNameTest(test_base.BaseTestCase):

    def test_mere_function(self):
        name = reflection.get_callable_name(mere_function)
        self.assertEqual('.'.join((__name__, 'mere_function')), name)

    def test_method(self):
        name = reflection.get_callable_name(Class.method)
        self.assertEqual('.'.join((__name__, 'Class', 'method')), name)

    def test_instance_method(self):
        name = reflection.get_callable_name(Class().method)
        self.assertEqual('.'.join((__name__, 'Class', 'method')), name)

    def test_static_method(self):
        name = reflection.get_callable_name(Class.static_method)
        self.assertEqual('.'.join((__name__, 'Class', 'static_method')), name)

    def test_class_method(self):
        name = reflection.get_callable_name(Class.class_method)
        self.assertEqual('.'.join((__name__, 'Class', 'class_method')), name)

    def test_constructor(self):
        name = reflection.get_callable_name(Class)
        self.assertEqual('.'.join((__name__, 'Class')), name)

    def test_callable_class(self):
        name = reflection.get_callable_name(CallableClass())
        self.assertEqual('.'.join((__name__, 'CallableClass')), name)

    def test_callable_class_call(self):
        name = reflection.get_callable_name(CallableClass().__call__)
        self.assertEqual('.'.join((__name__, 'CallableClass', '__call__')), name)