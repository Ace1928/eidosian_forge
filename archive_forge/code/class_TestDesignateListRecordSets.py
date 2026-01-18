from unittest import mock
from osc_lib.tests import utils
from designateclient.tests.osc import resources
from designateclient.v2 import base
from designateclient.v2.cli import recordsets
class TestDesignateListRecordSets(utils.TestCommand):

    def setUp(self):
        super().setUp()
        self.app.client_manager.dns = mock.MagicMock()
        self.cmd = recordsets.ListRecordSetsCommand(self.app, None)
        self.dns_client = self.app.client_manager.dns

    def test_list_recordsets(self):
        arg_list = ['6f106adb-0896-4114-b34f-4ac8dfee9465']
        verify_args = [('zone_id', '6f106adb-0896-4114-b34f-4ac8dfee9465')]
        body = resources.load('recordset_list')
        result = base.DesignateList()
        result.extend(body['recordsets'])
        self.dns_client.recordsets.list.return_value = result
        parsed_args = self.check_parser(self.cmd, arg_list, verify_args)
        columns, data = self.cmd.take_action(parsed_args)
        results = list(data)
        self.assertEqual(3, len(results))

    def test_list_all_recordsets(self):
        arg_list = ['all']
        verify_args = [('zone_id', 'all')]
        body = resources.load('recordset_list_all')
        result = base.DesignateList()
        result.extend(body['recordsets'])
        self.dns_client.recordsets.list_all_zones.return_value = result
        parsed_args = self.check_parser(self.cmd, arg_list, verify_args)
        columns, data = self.cmd.take_action(parsed_args)
        results = list(data)
        self.assertEqual(5, len(results))