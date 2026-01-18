import functools
from oslotest import base as test_base
from oslo_utils import reflection
class GetCallableArgsTest(test_base.BaseTestCase):

    def test_mere_function(self):
        result = reflection.get_callable_args(mere_function)
        self.assertEqual(['a', 'b'], result)

    def test_function_with_defaults(self):
        result = reflection.get_callable_args(function_with_defs)
        self.assertEqual(['a', 'b', 'optional'], result)

    def test_required_only(self):
        result = reflection.get_callable_args(function_with_defs, required_only=True)
        self.assertEqual(['a', 'b'], result)

    def test_method(self):
        result = reflection.get_callable_args(Class.method)
        self.assertEqual(['self', 'c', 'd'], result)

    def test_instance_method(self):
        result = reflection.get_callable_args(Class().method)
        self.assertEqual(['c', 'd'], result)

    def test_class_method(self):
        result = reflection.get_callable_args(Class.class_method)
        self.assertEqual(['g', 'h'], result)

    def test_class_constructor(self):
        result = reflection.get_callable_args(ClassWithInit)
        self.assertEqual(['k', 'lll'], result)

    def test_class_with_call(self):
        result = reflection.get_callable_args(CallableClass())
        self.assertEqual(['i', 'j'], result)

    def test_decorators_work(self):

        @dummy_decorator
        def special_fun(x, y):
            pass
        result = reflection.get_callable_args(special_fun)
        self.assertEqual(['x', 'y'], result)