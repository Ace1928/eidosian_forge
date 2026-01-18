import collections
import json
import os
from unittest import mock
import uuid
from heat.common import exception
from heat.common.i18n import _
from heat.common import identifier
from heat.common import template_format
from heat.common import urlfetch
from heat.engine import attributes
from heat.engine import environment
from heat.engine import properties
from heat.engine import resource
from heat.engine import resources
from heat.engine.resources import template_resource
from heat.engine import rsrc_defn
from heat.engine import stack as parser
from heat.engine import support
from heat.engine import template
from heat.objects import stack as stack_object
from heat.tests import common
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
class TemplateResourceCrudTest(common.HeatTestCase):
    provider = {'HeatTemplateFormatVersion': '2012-12-12', 'Parameters': {'Foo': {'Type': 'String'}, 'Blarg': {'Type': 'String', 'Default': 'wibble'}}}

    def setUp(self):
        super(TemplateResourceCrudTest, self).setUp()
        files = {'test_resource.template': json.dumps(self.provider)}
        self.ctx = utils.dummy_context()
        env = environment.Environment()
        env.load({'resource_registry': {'ResourceWithRequiredPropsAndEmptyAttrs': 'test_resource.template'}})
        self.stack = parser.Stack(self.ctx, 'test_stack', template.Template(empty_template, files=files, env=env), stack_id=str(uuid.uuid4()))
        self.defn = rsrc_defn.ResourceDefinition('test_t_res', 'ResourceWithRequiredPropsAndEmptyAttrs', {'Foo': 'bar'})
        self.res = template_resource.TemplateResource('test_t_res', self.defn, self.stack)
        self.assertIsNone(self.res.validate())
        self.patchobject(stack_object.Stack, 'get_status', return_value=('CREATE', 'COMPLETE', 'Created', 'Sometime'))

    def test_handle_create(self):
        self.res.create_with_template = mock.Mock(return_value=None)
        self.res.handle_create()
        self.res.create_with_template.assert_called_once_with(self.provider, {'Foo': 'bar'})

    def test_handle_adopt(self):
        self.res.create_with_template = mock.Mock(return_value=None)
        self.res.handle_adopt(resource_data={'resource_id': 'fred'})
        self.res.create_with_template.assert_called_once_with(self.provider, {'Foo': 'bar'}, adopt_data={'resource_id': 'fred'})

    def test_handle_update(self):
        self.res.update_with_template = mock.Mock(return_value=None)
        self.res.handle_update(self.defn, None, None)
        self.res.update_with_template.assert_called_once_with(self.provider, {'Foo': 'bar'})

    def test_handle_delete(self):
        self.res.rpc_client = mock.MagicMock()
        self.res.id = 55
        self.res.uuid = str(uuid.uuid4())
        self.res.resource_id = str(uuid.uuid4())
        self.res.action = self.res.CREATE
        self.res.nested = mock.MagicMock()
        ident = identifier.HeatIdentifier(self.ctx.tenant_id, self.res.physical_resource_name(), self.res.resource_id)
        self.res.nested().identifier.return_value = ident
        self.res.handle_delete()
        rpcc = self.res.rpc_client.return_value
        rpcc.delete_stack.assert_called_once_with(self.ctx, self.res.nested().identifier(), cast=False)