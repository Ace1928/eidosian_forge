from yaql.language import exceptions
from yaql.language import specs
from yaql.language import yaqltypes
import yaql.tests
class TestTypeAggregation(yaql.tests.TestCase):

    def test_not_of_type(self):

        @specs.parameter('arg', yaqltypes.NotOfType(int))
        def foo(arg):
            return True
        self.context.register_function(foo)
        self.assertTrue(self.eval('foo($)', data='abc'))
        self.assertTrue(self.eval('foo($)', data=[1, 2]))
        self.assertRaises(exceptions.NoMatchingFunctionException, self.eval, 'foo($)', data=123)
        self.assertRaises(exceptions.NoMatchingFunctionException, self.eval, 'foo($)', data=True)

    def test_chain(self):

        @specs.parameter('arg', yaqltypes.Chain(yaqltypes.NotOfType(bool), int))
        def foo(arg):
            return True
        self.context.register_function(foo)
        self.assertTrue(self.eval('foo($)', data=123))
        self.assertRaises(exceptions.NoMatchingFunctionException, self.eval, 'foo($)', data=True)
        self.assertRaises(exceptions.NoMatchingFunctionException, self.eval, 'foo($)', data='abc')

    def test_any_of(self):

        @specs.parameter('arg', yaqltypes.AnyOf(str, yaqltypes.Integer()))
        def foo(arg):
            if isinstance(arg, str):
                return 1
            if isinstance(arg, int):
                return 2
        self.context.register_function(foo)
        self.assertEqual(1, self.eval('foo($)', data='abc'))
        self.assertEqual(2, self.eval('foo($)', data=123))
        self.assertRaises(exceptions.NoMatchingFunctionException, self.eval, 'foo($)', data=True)
        self.assertRaises(exceptions.NoMatchingFunctionException, self.eval, 'foo($)', data=[1, 2])