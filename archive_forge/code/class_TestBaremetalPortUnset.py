import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from osc_lib import utils as oscutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_port
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
class TestBaremetalPortUnset(TestBaremetalPort):

    def setUp(self):
        super(TestBaremetalPortUnset, self).setUp()
        self.baremetal_mock.port.update.return_value = baremetal_fakes.FakeBaremetalResource(None, copy.deepcopy(baremetal_fakes.BAREMETAL_PORT), loaded=True)
        self.cmd = baremetal_port.UnsetBaremetalPort(self.app, None)

    def test_baremetal_port_unset_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(osctestutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_baremetal_port_unset_no_property(self):
        arglist = [baremetal_fakes.baremetal_port_uuid]
        verifylist = [('port', baremetal_fakes.baremetal_port_uuid)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.assertFalse(self.baremetal_mock.port.update.called)

    def test_baremetal_port_unset_extra(self):
        arglist = ['port', '--extra', 'foo']
        verifylist = [('port', 'port'), ('extra', ['foo'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.port.update.assert_called_once_with('port', [{'path': '/extra/foo', 'op': 'remove'}])

    def test_baremetal_port_unset_multiple_extras(self):
        arglist = ['port', '--extra', 'foo', '--extra', 'bar']
        verifylist = [('port', 'port'), ('extra', ['foo', 'bar'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.port.update.assert_called_once_with('port', [{'path': '/extra/foo', 'op': 'remove'}, {'path': '/extra/bar', 'op': 'remove'}])

    def test_baremetal_port_unset_portgroup_uuid(self):
        arglist = ['port', '--port-group']
        verifylist = [('port', 'port'), ('portgroup', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.port.update.assert_called_once_with('port', [{'path': '/portgroup_uuid', 'op': 'remove'}])

    def test_baremetal_port_unset_physical_network(self):
        arglist = ['port', '--physical-network']
        verifylist = [('port', 'port'), ('physical_network', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.port.update.assert_called_once_with('port', [{'path': '/physical_network', 'op': 'remove'}])

    def test_baremetal_port_unset_is_smartnic(self):
        arglist = ['port', '--is-smartnic']
        verifylist = [('port', 'port'), ('is_smartnic', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.port.update.assert_called_once_with('port', [{'path': '/is_smartnic', 'op': 'add', 'value': 'False'}])