from unittest import mock
from oslo_config import cfg
from requests import exceptions
import yaml
from heat.common import exception
from heat.common import identifier
from heat.common import template_format
from heat.common import urlfetch
from heat.engine import api
from heat.engine import node_data
from heat.engine import resource
from heat.engine.resources.aws.cfn import stack as stack_res
from heat.engine import rsrc_defn
from heat.engine import stack as parser
from heat.engine import template
from heat.objects import resource_data as resource_data_object
from heat.objects import stack as stack_object
from heat.tests import common
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
class NestedStackCrudTest(common.HeatTestCase):
    nested_template = "\nHeatTemplateFormatVersion: '2012-12-12'\nParameters:\n  KeyName:\n    Type: String\nOutputs:\n  Foo:\n    Value: bar\n"

    def setUp(self):
        super(NestedStackCrudTest, self).setUp()
        self.ctx = utils.dummy_context('test_username', 'aaaa', 'password')
        empty_template = {'HeatTemplateFormatVersion': '2012-12-12'}
        self.stack = parser.Stack(self.ctx, 'test', template.Template(empty_template))
        self.stack.store()
        self.patchobject(urlfetch, 'get', return_value=self.nested_template)
        self.nested_parsed = yaml.safe_load(self.nested_template)
        self.nested_params = {'KeyName': 'foo'}
        self.defn = rsrc_defn.ResourceDefinition('test_t_res', 'AWS::CloudFormation::Stack', {'TemplateURL': 'https://server.test/the.template', 'Parameters': self.nested_params})
        self.res = stack_res.NestedStack('test_t_res', self.defn, self.stack)
        self.assertIsNone(self.res.validate())
        self.res.store()
        self.patchobject(stack_object.Stack, 'get_status', return_value=('CREATE', 'COMPLETE', 'Created', 'Sometime'))

    def test_handle_create(self):
        self.res.create_with_template = mock.Mock(return_value=None)
        self.res.handle_create()
        self.res.create_with_template.assert_called_once_with(self.nested_parsed, self.nested_params, None, adopt_data=None)

    def test_handle_adopt(self):
        self.res.create_with_template = mock.Mock(return_value=None)
        self.res.handle_adopt(resource_data={'resource_id': 'fred'})
        self.res.create_with_template.assert_called_once_with(self.nested_parsed, self.nested_params, None, adopt_data={'resource_id': 'fred'})

    def test_handle_update(self):
        self.res.update_with_template = mock.Mock(return_value=None)
        self.res.handle_update(self.defn, None, None)
        self.res.update_with_template.assert_called_once_with(self.nested_parsed, self.nested_params, None)

    def test_handle_delete(self):
        self.res.rpc_client = mock.MagicMock()
        self.res.action = self.res.CREATE
        self.res.nested_identifier = mock.MagicMock()
        stack_identity = identifier.HeatIdentifier(self.ctx.tenant_id, self.res.physical_resource_name(), self.res.resource_id)
        self.res.nested_identifier.return_value = stack_identity
        self.res.resource_id = stack_identity.stack_id
        self.res.handle_delete()
        self.res.rpc_client.return_value.delete_stack.assert_called_once_with(self.ctx, stack_identity, cast=False)