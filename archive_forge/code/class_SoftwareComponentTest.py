import contextlib
from unittest import mock
from heat.common import exception as exc
from heat.common import template_format
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
class SoftwareComponentTest(common.HeatTestCase):

    def setUp(self):
        super(SoftwareComponentTest, self).setUp()
        self.ctx = utils.dummy_context()
        tpl = '\n        heat_template_version: 2013-05-23\n        resources:\n          mysql_component:\n            type: OS::Heat::SoftwareComponent\n            properties:\n              configs:\n                - actions: [CREATE]\n                  config: |\n                    #!/bin/bash\n                    echo "Create MySQL"\n                  tool: script\n                - actions: [UPDATE]\n                  config: |\n                    #!/bin/bash\n                    echo "Update MySQL"\n                  tool: script\n              inputs:\n                - name: mysql_port\n              outputs:\n                - name: root_password\n        '
        self.template = template_format.parse(tpl)
        self.stack = stack.Stack(self.ctx, 'software_component_test_stack', template.Template(self.template))
        self.component = self.stack['mysql_component']
        self.rpc_client = mock.MagicMock()
        self.component._rpc_client = self.rpc_client

        @contextlib.contextmanager
        def exc_filter(*args):
            try:
                yield
            except exc.NotFound:
                pass
        self.rpc_client.ignore_error_by_name.side_effect = exc_filter

    def test_handle_create(self):
        config_id = 'c8a19429-7fde-47ea-a42f-40045488226c'
        value = {'id': config_id}
        self.rpc_client.create_software_config.return_value = value
        props = dict(self.component.properties)
        self.component.handle_create()
        self.rpc_client.create_software_config.assert_called_with(self.ctx, group='component', name=None, inputs=props['inputs'], outputs=props['outputs'], config={'configs': props['configs']}, options=None)
        self.assertEqual(config_id, self.component.resource_id)

    def test_handle_delete(self):
        self.resource_id = None
        self.assertIsNone(self.component.handle_delete())
        config_id = 'c8a19429-7fde-47ea-a42f-40045488226c'
        self.component.resource_id = config_id
        self.rpc_client.delete_software_config.return_value = None
        self.assertIsNone(self.component.handle_delete())
        self.rpc_client.delete_software_config.side_effect = exc.NotFound
        self.assertIsNone(self.component.handle_delete())

    def test_resolve_attribute(self):
        self.assertIsNone(self.component._resolve_attribute('others'))
        self.component.resource_id = None
        self.assertIsNone(self.component._resolve_attribute('configs'))
        self.component.resource_id = 'c8a19429-7fde-47ea-a42f-40045488226c'
        configs = self.template['resources']['mysql_component']['properties']['configs']
        value = {'config': {'configs': configs}}
        self.rpc_client.show_software_config.return_value = value
        self.assertEqual(configs, self.component._resolve_attribute('configs'))
        self.rpc_client.show_software_config.side_effect = exc.NotFound
        self.assertIsNone(self.component._resolve_attribute('configs'))