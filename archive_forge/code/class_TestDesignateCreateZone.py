from unittest import mock
from osc_lib.tests import utils
from designateclient.tests.osc import resources
from designateclient.v2 import base
from designateclient.v2.cli import zones
class TestDesignateCreateZone(utils.TestCommand):

    def setUp(self):
        super().setUp()
        self.app.client_manager.dns = mock.MagicMock()
        self.cmd = zones.CreateZoneCommand(self.app, None)
        self.dns_client = self.app.client_manager.dns

    def test_create_zone(self):
        arg_list = ['example.devstack.org.', '--email', 'fake@devstack.org']
        verify_args = [('name', 'example.devstack.org.'), ('email', 'fake@devstack.org')]
        body = resources.load('zone_create')
        self.dns_client.zones.create.return_value = body
        parsed_args = self.check_parser(self.cmd, arg_list, verify_args)
        columns, data = self.cmd.take_action(parsed_args)
        results = list(data)
        self.assertEqual(17, len(results))