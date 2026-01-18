import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient.osc.v1 import baremetal_portgroup
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
class TestBaremetalPortGroupUnset(TestBaremetalPortGroup):

    def setUp(self):
        super(TestBaremetalPortGroupUnset, self).setUp()
        self.baremetal_mock.portgroup.update.return_value = baremetal_fakes.FakeBaremetalResource(None, copy.deepcopy(baremetal_fakes.PORTGROUP), loaded=True)
        self.cmd = baremetal_portgroup.UnsetBaremetalPortGroup(self.app, None)

    def test_baremetal_portgroup_unset_extra(self):
        arglist = ['portgroup', '--extra', 'key1']
        verifylist = [('portgroup', 'portgroup'), ('extra', ['key1'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.portgroup.update.assert_called_once_with('portgroup', [{'path': '/extra/key1', 'op': 'remove'}])

    def test_baremetal_portgroup_unset_name(self):
        arglist = ['portgroup', '--name']
        verifylist = [('name', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.portgroup.update.assert_called_once_with('portgroup', [{'path': '/name', 'op': 'remove'}])

    def test_baremetal_portgroup_unset_address(self):
        arglist = ['portgroup', '--address']
        verifylist = [('address', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.portgroup.update.assert_called_once_with('portgroup', [{'path': '/address', 'op': 'remove'}])

    def test_baremetal_portgroup_unset_multiple_extras(self):
        arglist = ['portgroup', '--extra', 'key1', '--extra', 'key2']
        verifylist = [('portgroup', 'portgroup'), ('extra', ['key1', 'key2'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.portgroup.update.assert_called_once_with('portgroup', [{'path': '/extra/key1', 'op': 'remove'}, {'path': '/extra/key2', 'op': 'remove'}])

    def test_baremetal_portgroup_unset_multiple_properties(self):
        arglist = ['portgroup', '--property', 'key1', '--property', 'key2']
        verifylist = [('portgroup', 'portgroup'), ('properties', ['key1', 'key2'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.baremetal_mock.portgroup.update.assert_called_once_with('portgroup', [{'path': '/properties/key1', 'op': 'remove'}, {'path': '/properties/key2', 'op': 'remove'}])

    def test_baremetal_portgroup_unset_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(osctestutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_baremetal_portgroup_unset_no_property(self):
        uuid = baremetal_fakes.baremetal_portgroup_uuid
        arglist = [uuid]
        verifylist = [('portgroup', uuid)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.assertFalse(self.baremetal_mock.portgroup.update.called)