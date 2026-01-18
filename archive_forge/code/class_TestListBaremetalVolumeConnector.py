import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_volume_connector as bm_vol_connector
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
class TestListBaremetalVolumeConnector(TestBaremetalVolumeConnector):

    def setUp(self):
        super(TestListBaremetalVolumeConnector, self).setUp()
        self.baremetal_mock.volume_connector.list.return_value = [baremetal_fakes.FakeBaremetalResource(None, copy.deepcopy(baremetal_fakes.VOLUME_CONNECTOR), loaded=True)]
        self.cmd = bm_vol_connector.ListBaremetalVolumeConnector(self.app, None)

    def test_baremetal_volume_connector_list(self):
        arglist = []
        verifylist = []
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'marker': None, 'limit': None}
        self.baremetal_mock.volume_connector.list.assert_called_once_with(**kwargs)
        collist = ('UUID', 'Node UUID', 'Type', 'Connector ID')
        self.assertEqual(collist, columns)
        datalist = ((baremetal_fakes.baremetal_volume_connector_uuid, baremetal_fakes.baremetal_uuid, baremetal_fakes.baremetal_volume_connector_type, baremetal_fakes.baremetal_volume_connector_connector_id),)
        self.assertEqual(datalist, tuple(data))

    def test_baremetal_volume_connector_list_node(self):
        arglist = ['--node', baremetal_fakes.baremetal_uuid]
        verifylist = [('node', baremetal_fakes.baremetal_uuid)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'node': baremetal_fakes.baremetal_uuid, 'marker': None, 'limit': None}
        self.baremetal_mock.volume_connector.list.assert_called_once_with(**kwargs)
        collist = ('UUID', 'Node UUID', 'Type', 'Connector ID')
        self.assertEqual(collist, columns)
        datalist = ((baremetal_fakes.baremetal_volume_connector_uuid, baremetal_fakes.baremetal_uuid, baremetal_fakes.baremetal_volume_connector_type, baremetal_fakes.baremetal_volume_connector_connector_id),)
        self.assertEqual(datalist, tuple(data))

    def test_baremetal_volume_connector_list_long(self):
        arglist = ['--long']
        verifylist = [('detail', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'detail': True, 'marker': None, 'limit': None}
        self.baremetal_mock.volume_connector.list.assert_called_with(**kwargs)
        collist = ('UUID', 'Node UUID', 'Type', 'Connector ID', 'Extra', 'Created At', 'Updated At')
        self.assertEqual(collist, columns)
        datalist = ((baremetal_fakes.baremetal_volume_connector_uuid, baremetal_fakes.baremetal_uuid, baremetal_fakes.baremetal_volume_connector_type, baremetal_fakes.baremetal_volume_connector_connector_id, baremetal_fakes.baremetal_volume_connector_extra, '', ''),)
        self.assertEqual(datalist, tuple(data))

    def test_baremetal_volume_connector_list_fields(self):
        arglist = ['--fields', 'uuid', 'connector_id']
        verifylist = [('fields', [['uuid', 'connector_id']])]
        fake_vc = copy.deepcopy(baremetal_fakes.VOLUME_CONNECTOR)
        fake_vc.pop('type')
        fake_vc.pop('extra')
        fake_vc.pop('node_uuid')
        self.baremetal_mock.volume_connector.list.return_value = [baremetal_fakes.FakeBaremetalResource(None, fake_vc, loaded=True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'detail': False, 'marker': None, 'limit': None, 'fields': ('uuid', 'connector_id')}
        self.baremetal_mock.volume_connector.list.assert_called_with(**kwargs)
        collist = ('UUID', 'Connector ID')
        self.assertEqual(collist, columns)
        datalist = ((baremetal_fakes.baremetal_volume_connector_uuid, baremetal_fakes.baremetal_volume_connector_connector_id),)
        self.assertEqual(datalist, tuple(data))

    def test_baremetal_volume_connector_list_fields_multiple(self):
        arglist = ['--fields', 'uuid', 'connector_id', '--fields', 'extra']
        verifylist = [('fields', [['uuid', 'connector_id'], ['extra']])]
        fake_vc = copy.deepcopy(baremetal_fakes.VOLUME_CONNECTOR)
        fake_vc.pop('type')
        fake_vc.pop('node_uuid')
        self.baremetal_mock.volume_connector.list.return_value = [baremetal_fakes.FakeBaremetalResource(None, fake_vc, loaded=True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        kwargs = {'detail': False, 'marker': None, 'limit': None, 'fields': ('uuid', 'connector_id', 'extra')}
        self.baremetal_mock.volume_connector.list.assert_called_with(**kwargs)
        collist = ('UUID', 'Connector ID', 'Extra')
        self.assertEqual(collist, columns)
        datalist = ((baremetal_fakes.baremetal_volume_connector_uuid, baremetal_fakes.baremetal_volume_connector_connector_id, baremetal_fakes.baremetal_volume_connector_extra),)
        self.assertEqual(datalist, tuple(data))

    def test_baremetal_volume_connector_list_invalid_fields(self):
        arglist = ['--fields', 'uuid', 'invalid']
        verifylist = [('fields', [['uuid', 'invalid']])]
        self.assertRaises(osctestutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_baremetal_volume_connector_list_marker(self):
        arglist = ['--marker', baremetal_fakes.baremetal_volume_connector_uuid]
        verifylist = [('marker', baremetal_fakes.baremetal_volume_connector_uuid)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        kwargs = {'marker': baremetal_fakes.baremetal_volume_connector_uuid, 'limit': None}
        self.baremetal_mock.volume_connector.list.assert_called_once_with(**kwargs)

    def test_baremetal_volume_connector_list_limit(self):
        arglist = ['--limit', '10']
        verifylist = [('limit', 10)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        kwargs = {'marker': None, 'limit': 10}
        self.baremetal_mock.volume_connector.list.assert_called_once_with(**kwargs)

    def test_baremetal_volume_connector_list_sort(self):
        arglist = ['--sort', 'type']
        verifylist = [('sort', 'type')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        kwargs = {'marker': None, 'limit': None}
        self.baremetal_mock.volume_connector.list.assert_called_once_with(**kwargs)

    def test_baremetal_volume_connector_list_sort_desc(self):
        arglist = ['--sort', 'type:desc']
        verifylist = [('sort', 'type:desc')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        kwargs = {'marker': None, 'limit': None}
        self.baremetal_mock.volume_connector.list.assert_called_once_with(**kwargs)

    def test_baremetal_volume_connector_list_exclusive_options(self):
        arglist = ['--fields', 'uuid', '--long']
        self.assertRaises(osctestutils.ParserException, self.check_parser, self.cmd, arglist, [])

    def test_baremetal_volume_connector_list_negative_limit(self):
        arglist = ['--limit', '-1']
        verifylist = [('limit', -1)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.assertRaises(exc.CommandError, self.cmd.take_action, parsed_args)