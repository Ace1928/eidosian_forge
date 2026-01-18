from unittest import mock
from osc_lib.tests import utils
from designateclient.tests.osc import resources
from designateclient.v2 import base
from designateclient.v2.cli import zones
class TestDesignateListZones(utils.TestCommand):

    def setUp(self):
        super().setUp()
        self.app.client_manager.dns = mock.MagicMock()
        self.cmd = zones.ListZonesCommand(self.app, None)
        self.dns_client = self.app.client_manager.dns

    def test_list_zones(self):
        arg_list = []
        verify_args = []
        body = resources.load('zone_list')
        result = base.DesignateList()
        result.extend(body['zones'])
        self.dns_client.zones.list.return_value = result
        parsed_args = self.check_parser(self.cmd, arg_list, verify_args)
        columns, data = self.cmd.take_action(parsed_args)
        results = list(data)
        self.assertEqual(2, len(results))