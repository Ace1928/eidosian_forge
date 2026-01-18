import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_volume_connector as bm_vol_connector
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
class TestCreateBaremetalVolumeConnector(TestBaremetalVolumeConnector):

    def setUp(self):
        super(TestCreateBaremetalVolumeConnector, self).setUp()
        self.baremetal_mock.volume_connector.create.return_value = baremetal_fakes.FakeBaremetalResource(None, copy.deepcopy(baremetal_fakes.VOLUME_CONNECTOR), loaded=True)
        self.cmd = bm_vol_connector.CreateBaremetalVolumeConnector(self.app, None)

    def test_baremetal_volume_connector_create(self):
        arglist = ['--node', baremetal_fakes.baremetal_uuid, '--type', baremetal_fakes.baremetal_volume_connector_type, '--connector-id', baremetal_fakes.baremetal_volume_connector_connector_id, '--uuid', baremetal_fakes.baremetal_volume_connector_uuid]
        verifylist = [('node_uuid', baremetal_fakes.baremetal_uuid), ('type', baremetal_fakes.baremetal_volume_connector_type), ('connector_id', baremetal_fakes.baremetal_volume_connector_connector_id), ('uuid', baremetal_fakes.baremetal_volume_connector_uuid)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        args = {'node_uuid': baremetal_fakes.baremetal_uuid, 'type': baremetal_fakes.baremetal_volume_connector_type, 'connector_id': baremetal_fakes.baremetal_volume_connector_connector_id, 'uuid': baremetal_fakes.baremetal_volume_connector_uuid}
        self.baremetal_mock.volume_connector.create.assert_called_once_with(**args)

    def test_baremetal_volume_connector_create_without_uuid(self):
        arglist = ['--node', baremetal_fakes.baremetal_uuid, '--type', baremetal_fakes.baremetal_volume_connector_type, '--connector-id', baremetal_fakes.baremetal_volume_connector_connector_id]
        verifylist = [('node_uuid', baremetal_fakes.baremetal_uuid), ('type', baremetal_fakes.baremetal_volume_connector_type), ('connector_id', baremetal_fakes.baremetal_volume_connector_connector_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        args = {'node_uuid': baremetal_fakes.baremetal_uuid, 'type': baremetal_fakes.baremetal_volume_connector_type, 'connector_id': baremetal_fakes.baremetal_volume_connector_connector_id}
        self.baremetal_mock.volume_connector.create.assert_called_once_with(**args)

    def test_baremetal_volume_connector_create_extras(self):
        arglist = ['--node', baremetal_fakes.baremetal_uuid, '--type', baremetal_fakes.baremetal_volume_connector_type, '--connector-id', baremetal_fakes.baremetal_volume_connector_connector_id, '--extra', 'key1=value1', '--extra', 'key2=value2']
        verifylist = [('node_uuid', baremetal_fakes.baremetal_uuid), ('type', baremetal_fakes.baremetal_volume_connector_type), ('connector_id', baremetal_fakes.baremetal_volume_connector_connector_id), ('extra', ['key1=value1', 'key2=value2'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        args = {'node_uuid': baremetal_fakes.baremetal_uuid, 'type': baremetal_fakes.baremetal_volume_connector_type, 'connector_id': baremetal_fakes.baremetal_volume_connector_connector_id, 'extra': baremetal_fakes.baremetal_volume_connector_extra}
        self.baremetal_mock.volume_connector.create.assert_called_once_with(**args)

    def test_baremetal_volume_connector_create_invalid_type(self):
        arglist = ['--node', baremetal_fakes.baremetal_uuid, '--type', 'invalid', '--connector-id', baremetal_fakes.baremetal_volume_connector_connector_id, '--uuid', baremetal_fakes.baremetal_volume_connector_uuid]
        verifylist = None
        self.assertRaises(osctestutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_baremetal_volume_connector_create_missing_node(self):
        arglist = ['--type', baremetal_fakes.baremetal_volume_connector_type, '--connector-id', baremetal_fakes.baremetal_volume_connector_connector_id, '--uuid', baremetal_fakes.baremetal_volume_connector_uuid]
        verifylist = None
        self.assertRaises(osctestutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_baremetal_volume_connector_create_missing_type(self):
        arglist = ['--node', baremetal_fakes.baremetal_uuid, '--connector-id', baremetal_fakes.baremetal_volume_connector_connector_id, '--uuid', baremetal_fakes.baremetal_volume_connector_uuid]
        verifylist = None
        self.assertRaises(osctestutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_baremetal_volume_connector_create_missing_connector_id(self):
        arglist = ['--node', baremetal_fakes.baremetal_uuid, '--type', baremetal_fakes.baremetal_volume_connector_type, '--uuid', baremetal_fakes.baremetal_volume_connector_uuid]
        verifylist = None
        self.assertRaises(osctestutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)