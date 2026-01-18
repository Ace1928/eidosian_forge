import copy
import hashlib
import json
import fixtures
from stevedore import extension
from heat.common import exception
from heat.common import template_format
from heat.engine.cfn import functions as cfn_funcs
from heat.engine.cfn import parameters as cfn_p
from heat.engine.cfn import template as cfn_t
from heat.engine.clients.os import nova
from heat.engine import environment
from heat.engine import function
from heat.engine.hot import template as hot_t
from heat.engine import node_data
from heat.engine import rsrc_defn
from heat.engine import stack
from heat.engine import stk_defn
from heat.engine import template
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
class ResolveDataTest(common.HeatTestCase):

    def setUp(self):
        super(ResolveDataTest, self).setUp()
        self.username = 'parser_stack_test_user'
        self.ctx = utils.dummy_context()
        self.stack = stack.Stack(self.ctx, 'resolve_test_stack', template.Template(empty_template))

    def resolve(self, snippet):
        return function.resolve(self.stack.t.parse(self.stack.defn, snippet))

    def test_join_split(self):
        snippet = {'Fn::Join': [';', ['one', 'two', 'three']]}
        self.assertEqual('one;two;three', self.resolve(snippet))
        snippet = {'Fn::Split': [';', snippet]}
        self.assertEqual(['one', 'two', 'three'], self.resolve(snippet))

    def test_split_join_split_join(self):
        snippet = {'Fn::Split': [',', 'one,two,three']}
        self.assertEqual(['one', 'two', 'three'], self.resolve(snippet))
        snippet = {'Fn::Join': [';', snippet]}
        self.assertEqual('one;two;three', self.resolve(snippet))
        snippet = {'Fn::Split': [';', snippet]}
        self.assertEqual(['one', 'two', 'three'], self.resolve(snippet))
        snippet = {'Fn::Join': ['-', snippet]}
        self.assertEqual('one-two-three', self.resolve(snippet))

    def test_join_recursive(self):
        raw = {'Fn::Join': ['\n', [{'Fn::Join': [' ', ['foo', 'bar']]}, 'baz']]}
        self.assertEqual('foo bar\nbaz', self.resolve(raw))

    def test_join_not_string(self):
        snippet = {'Fn::Join': ['\n', [{'Fn::Join': [' ', ['foo', 45]]}, 'baz']]}
        error = self.assertRaises(TypeError, self.resolve, snippet)
        self.assertIn('45', str(error))

    def test_base64_replace(self):
        raw = {'Fn::Base64': {'Fn::Replace': [{'foo': 'bar'}, 'Meet at the foo']}}
        self.assertEqual('Meet at the bar', self.resolve(raw))

    def test_replace_base64(self):
        raw = {'Fn::Replace': [{'foo': 'bar'}, {'Fn::Base64': 'Meet at the foo'}]}
        self.assertEqual('Meet at the bar', self.resolve(raw))

    def test_nested_selects(self):
        data = {'a': ['one', 'two', 'three'], 'b': ['een', 'twee', {'d': 'D', 'e': 'E'}]}
        raw = {'Fn::Select': ['a', data]}
        self.assertEqual(data['a'], self.resolve(raw))
        raw = {'Fn::Select': ['b', data]}
        self.assertEqual(data['b'], self.resolve(raw))
        raw = {'Fn::Select': ['1', {'Fn::Select': ['b', data]}]}
        self.assertEqual('twee', self.resolve(raw))
        raw = {'Fn::Select': ['e', {'Fn::Select': ['2', {'Fn::Select': ['b', data]}]}]}
        self.assertEqual('E', self.resolve(raw))

    def test_member_list_select(self):
        snippet = {'Fn::Select': ['metric', {'Fn::MemberListToMap': ['Name', 'Value', ['.member.0.Name=metric', '.member.0.Value=cpu', '.member.1.Name=size', '.member.1.Value=56']]}]}
        self.assertEqual('cpu', self.resolve(snippet))