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
class ValidateTest(common.HeatTestCase):

    def setUp(self):
        super(ValidateTest, self).setUp()
        self.func = TestFunction(None, 'foo', ['bar', 'baz'])

    def test_validate_func(self):
        self.assertIsNone(function.validate(self.func))
        self.func = TestFunction(None, 'foo', ['bar'])
        self.assertRaisesRegex(exception.StackValidationFailed, 'foo: Need more arguments', function.validate, self.func)

    def test_validate_dict(self):
        snippet = {'foo': 'bar', 'blarg': self.func}
        function.validate(snippet)
        self.func = TestFunction(None, 'foo', ['bar'])
        snippet = {'foo': 'bar', 'blarg': self.func}
        self.assertRaisesRegex(exception.StackValidationFailed, 'blarg.foo: Need more arguments', function.validate, snippet)

    def test_validate_list(self):
        snippet = ['foo', 'bar', 'baz', 'blarg', self.func]
        function.validate(snippet)
        self.func = TestFunction(None, 'foo', ['bar'])
        snippet = {'foo': 'bar', 'blarg': self.func}
        self.assertRaisesRegex(exception.StackValidationFailed, 'blarg.foo: Need more arguments', function.validate, snippet)

    def test_validate_all(self):
        snippet = ['foo', {'bar': ['baz', {'blarg': self.func}]}]
        function.validate(snippet)
        self.func = TestFunction(None, 'foo', ['bar'])
        snippet = {'foo': 'bar', 'blarg': self.func}
        self.assertRaisesRegex(exception.StackValidationFailed, 'blarg.foo: Need more arguments', function.validate, snippet)