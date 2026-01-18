from unittest import mock
from heat.common import exception
from heat.engine.resources.openstack.heat import structured_config as sc
from heat.engine import rsrc_defn
from heat.engine import software_config_io as swc_io
from heat.engine import stack as parser
from heat.engine import template
from heat.rpc import api as rpc_api
from heat.tests import common
from heat.tests import utils
class StructuredConfigTestJSON(common.HeatTestCase):
    template = {'HeatTemplateFormatVersion': '2012-12-12', 'Resources': {'config_mysql': {'Type': 'OS::Heat::StructuredConfig', 'Properties': {'config': {'foo': 'bar'}}}}}
    stored_config = {'foo': 'bar'}

    def setUp(self):
        super(StructuredConfigTestJSON, self).setUp()
        self.ctx = utils.dummy_context()
        self.properties = {'config': {'foo': 'bar'}}
        self.stack = parser.Stack(self.ctx, 'software_config_test_stack', template.Template(self.template))
        self.config = self.stack['config_mysql']
        self.rpc_client = mock.MagicMock()
        self.config._rpc_client = self.rpc_client

    def test_handle_create(self):
        config_id = 'c8a19429-7fde-47ea-a42f-40045488226c'
        value = {'id': config_id}
        self.rpc_client.create_software_config.return_value = value
        self.config.handle_create()
        self.assertEqual(config_id, self.config.resource_id)
        kwargs = self.rpc_client.create_software_config.call_args[1]
        self.assertEqual(self.stored_config, kwargs['config'])