import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_volume_connector as bm_vol_connector
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
class TestSetBaremetalVolumeConnector(TestBaremetalVolumeConnector):

    def setUp(self):
        super(TestSetBaremetalVolumeConnector, self).setUp()
        self.cmd = bm_vol_connector.SetBaremetalVolumeConnector(self.app, None)

    def test_baremetal_volume_connector_set_node_uuid(self):
        new_node_uuid = 'xxx-xxxxxx-zzzz'
        arglist = [baremetal_fakes.baremetal_volume_connector_uuid, '--node', new_node_uuid]
        verifylist = [('volume_connector', baremetal_fakes.baremetal_volume_connector_uuid), ('node_uuid', new_node_uuid)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.volume_connector.update.assert_called_once_with(baremetal_fakes.baremetal_volume_connector_uuid, [{'path': '/node_uuid', 'value': new_node_uuid, 'op': 'add'}])

    def test_baremetal_volume_connector_set_type(self):
        new_type = 'wwnn'
        arglist = [baremetal_fakes.baremetal_volume_connector_uuid, '--type', new_type]
        verifylist = [('volume_connector', baremetal_fakes.baremetal_volume_connector_uuid), ('type', new_type)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.volume_connector.update.assert_called_once_with(baremetal_fakes.baremetal_volume_connector_uuid, [{'path': '/type', 'value': new_type, 'op': 'add'}])

    def test_baremetal_volume_connector_set_invalid_type(self):
        new_type = 'invalid'
        arglist = [baremetal_fakes.baremetal_volume_connector_uuid, '--type', new_type]
        verifylist = None
        self.assertRaises(osctestutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_baremetal_volume_connector_set_connector_id(self):
        new_conn_id = '11:22:33:44:55:66:77:88'
        arglist = [baremetal_fakes.baremetal_volume_connector_uuid, '--connector-id', new_conn_id]
        verifylist = [('volume_connector', baremetal_fakes.baremetal_volume_connector_uuid), ('connector_id', new_conn_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.volume_connector.update.assert_called_once_with(baremetal_fakes.baremetal_volume_connector_uuid, [{'path': '/connector_id', 'value': new_conn_id, 'op': 'add'}])

    def test_baremetal_volume_connector_set_type_and_connector_id(self):
        new_type = 'wwnn'
        new_conn_id = '11:22:33:44:55:66:77:88'
        arglist = [baremetal_fakes.baremetal_volume_connector_uuid, '--type', new_type, '--connector-id', new_conn_id]
        verifylist = [('volume_connector', baremetal_fakes.baremetal_volume_connector_uuid), ('type', new_type), ('connector_id', new_conn_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.volume_connector.update.assert_called_once_with(baremetal_fakes.baremetal_volume_connector_uuid, [{'path': '/type', 'value': new_type, 'op': 'add'}, {'path': '/connector_id', 'value': new_conn_id, 'op': 'add'}])

    def test_baremetal_volume_connector_set_extra(self):
        arglist = [baremetal_fakes.baremetal_volume_connector_uuid, '--extra', 'foo=bar']
        verifylist = [('volume_connector', baremetal_fakes.baremetal_volume_connector_uuid), ('extra', ['foo=bar'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.volume_connector.update.assert_called_once_with(baremetal_fakes.baremetal_volume_connector_uuid, [{'path': '/extra/foo', 'value': 'bar', 'op': 'add'}])

    def test_baremetal_volume_connector_set_multiple_extras(self):
        arglist = [baremetal_fakes.baremetal_volume_connector_uuid, '--extra', 'key1=val1', '--extra', 'key2=val2']
        verifylist = [('volume_connector', baremetal_fakes.baremetal_volume_connector_uuid), ('extra', ['key1=val1', 'key2=val2'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.volume_connector.update.assert_called_once_with(baremetal_fakes.baremetal_volume_connector_uuid, [{'path': '/extra/key1', 'value': 'val1', 'op': 'add'}, {'path': '/extra/key2', 'value': 'val2', 'op': 'add'}])

    def test_baremetal_volume_connector_set_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(osctestutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_baremetal_volume_connector_set_no_property(self):
        arglist = [baremetal_fakes.baremetal_volume_connector_uuid]
        verifylist = [('volume_connector', baremetal_fakes.baremetal_volume_connector_uuid)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.volume_connector.update.assert_not_called()