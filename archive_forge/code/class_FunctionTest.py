import copy
import uuid
from heat.common import exception
from heat.common.i18n import _
from heat.engine.cfn import functions
from heat.engine import environment
from heat.engine import function
from heat.engine import resource
from heat.engine import rsrc_defn
from heat.engine import stack
from heat.engine import stk_defn
from heat.engine import template
from heat.tests import common
from heat.tests import utils
class FunctionTest(common.HeatTestCase):

    def test_equal(self):
        func = TestFunction(None, 'foo', ['bar', 'baz'])
        self.assertTrue(func == 'wibble')
        self.assertTrue('wibble' == func)

    def test_not_equal(self):
        func = TestFunction(None, 'foo', ['bar', 'baz'])
        self.assertTrue(func != 'foo')
        self.assertTrue('foo' != func)

    def test_equal_func(self):
        func1 = TestFunction(None, 'foo', ['bar', 'baz'])
        func2 = TestFunction(None, 'blarg', ['wibble', 'quux'])
        self.assertTrue(func1 == func2)

    def test_function_str_value(self):
        func1 = TestFunction(None, 'foo', ['bar', 'baz'])
        expected = '%s %s' % ('<heat.tests.test_function.TestFunction', "{foo: ['bar', 'baz']} -> 'wibble'>")
        self.assertEqual(expected, str(func1))

    def test_function_stack_reference_none(self):
        func1 = TestFunction(None, 'foo', ['bar', 'baz'])
        self.assertIsNone(func1.stack)

    def test_function_exception_key_error(self):
        func1 = TestFunctionKeyError(None, 'foo', ['bar', 'baz'])
        expected = '%s %s' % ('<heat.tests.test_function.TestFunctionKeyError', "{foo: ['bar', 'baz']} -> ???>")
        self.assertEqual(expected, str(func1))

    def test_function_eq_exception_key_error(self):
        func1 = TestFunctionKeyError(None, 'foo', ['bar', 'baz'])
        func2 = TestFunctionKeyError(None, 'foo', ['bar', 'baz'])
        result = func1.__eq__(func2)
        self.assertEqual(result, NotImplemented)

    def test_function_ne_exception_key_error(self):
        func1 = TestFunctionKeyError(None, 'foo', ['bar', 'baz'])
        func2 = TestFunctionKeyError(None, 'foo', ['bar', 'baz'])
        result = func1.__ne__(func2)
        self.assertEqual(result, NotImplemented)

    def test_function_exception_value_error(self):
        func1 = TestFunctionValueError(None, 'foo', ['bar', 'baz'])
        expected = '%s %s' % ('<heat.tests.test_function.TestFunctionValueError', "{foo: ['bar', 'baz']} -> ???>")
        self.assertEqual(expected, str(func1))

    def test_function_eq_exception_value_error(self):
        func1 = TestFunctionValueError(None, 'foo', ['bar', 'baz'])
        func2 = TestFunctionValueError(None, 'foo', ['bar', 'baz'])
        result = func1.__eq__(func2)
        self.assertEqual(result, NotImplemented)

    def test_function_ne_exception_value_error(self):
        func1 = TestFunctionValueError(None, 'foo', ['bar', 'baz'])
        func2 = TestFunctionValueError(None, 'foo', ['bar', 'baz'])
        result = func1.__ne__(func2)
        self.assertEqual(result, NotImplemented)

    def test_function_abstract_result(self):
        func1 = TestFunctionResult(None, 'foo', ['bar', 'baz'])
        expected = '%s %s -> %s' % ('<heat.tests.test_function.TestFunctionResult', "{foo: ['bar', 'baz']}", "{'foo': ['bar', 'baz']}>")
        self.assertEqual(expected, str(func1))

    def test_copy(self):
        func = TestFunction(None, 'foo', ['bar', 'baz'])
        self.assertEqual({'foo': ['bar', 'baz']}, copy.deepcopy(func))