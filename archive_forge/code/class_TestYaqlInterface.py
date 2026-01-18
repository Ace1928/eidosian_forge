from yaql.language import specs
from yaql.language import yaqltypes
import yaql.tests
class TestYaqlInterface(yaql.tests.TestCase):

    def test_call(self):

        def foo(yaql_interface):
            return yaql_interface('2+2')

        @specs.inject('yi', yaqltypes.YaqlInterface())
        def bar(yi):
            return yi('$a * $', 2, a=3)
        self.context.register_function(foo)
        self.context.register_function(bar)
        self.assertEqual(4, self.eval('foo()'))
        self.assertEqual(6, self.eval('bar()'))

    def test_function_call(self):

        def foo(yaql_interface):
            return yaql_interface.len([1, 2, 3])
        self.context.register_function(foo)
        self.assertEqual(3, self.eval('foo()'))

    def test_method_call(self):

        def foo(yaql_interface):
            return yaql_interface.on([1, 2, 3]).where(lambda i: i > 1)

        @specs.inject('yi', yaqltypes.YaqlInterface())
        def bar(yi):
            return yi.on([1, 2, 3]).select(yi.engine('$ * $'))
        self.context.register_function(foo)
        self.context.register_function(bar)
        self.assertEqual([2, 3], self.eval('foo()'))
        self.assertEqual([1, 4, 9], self.eval('bar()'))

    def test_data_access(self):

        def foo(yaql_interface):
            return (yaql_interface[''], yaql_interface['key'])
        self.context.register_function(foo)
        self.context['key'] = 'value'
        self.assertEqual(['test', 'value'], self.eval('foo()', data='test'))