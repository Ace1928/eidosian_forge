import contextlib
from unittest import mock
from heat.common import exception as exc
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
class SoftwareConfigTest(common.HeatTestCase):

    def setUp(self):
        super(SoftwareConfigTest, self).setUp()
        self.ctx = utils.dummy_context()
        self.properties = {'group': 'Heat::Shell', 'inputs': [], 'outputs': [], 'options': {}, 'config': '#!/bin/bash'}
        self.stack = stack.Stack(self.ctx, 'software_config_test_stack', template.Template({'HeatTemplateFormatVersion': '2012-12-12', 'Resources': {'config_mysql': {'Type': 'OS::Heat::SoftwareConfig', 'Properties': self.properties}}}))
        self.config = self.stack['config_mysql']
        self.rpc_client = mock.MagicMock()
        self.config._rpc_client = self.rpc_client

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
        self.config.handle_create()
        self.assertEqual(config_id, self.config.resource_id)

    def test_handle_delete(self):
        self.resource_id = None
        self.assertIsNone(self.config.handle_delete())
        config_id = 'c8a19429-7fde-47ea-a42f-40045488226c'
        self.config.resource_id = config_id
        self.rpc_client.delete_software_config.return_value = None
        self.assertIsNone(self.config.handle_delete())
        self.rpc_client.delete_software_config.side_effect = exc.NotFound
        self.assertIsNone(self.config.handle_delete())

    def test_resolve_attribute(self):
        self.assertIsNone(self.config._resolve_attribute('others'))
        self.config.resource_id = None
        self.assertIsNone(self.config._resolve_attribute('config'))
        self.config.resource_id = 'c8a19429-7fde-47ea-a42f-40045488226c'
        value = {'config': '#!/bin/bash'}
        self.rpc_client.show_software_config.return_value = value
        self.assertEqual('#!/bin/bash', self.config._resolve_attribute('config'))
        self.rpc_client.show_software_config.side_effect = exc.NotFound
        self.assertIsNone(self.config._resolve_attribute('config'))