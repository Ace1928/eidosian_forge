import copy
from unittest import mock
from unittest.mock import call
from magnumclient.osc.v1 import nodegroups as osc_nodegroups
from magnumclient.tests.osc.unit.v1 import fakes as magnum_fakes
class TestNodeGroupDelete(TestNodeGroup):

    def setUp(self):
        super(TestNodeGroupDelete, self).setUp()
        self.ng_mock.delete = mock.Mock()
        self.ng_mock.delete.return_value = None
        self.cmd = osc_nodegroups.DeleteNodeGroup(self.app, None)

    def test_nodegroup_delete_one(self):
        arglist = ['foo', 'fake-nodegroup']
        verifylist = [('cluster', 'foo'), ('nodegroup', ['fake-nodegroup'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.ng_mock.delete.assert_called_with('foo', 'fake-nodegroup')

    def test_nodegroup_delete_multiple(self):
        arglist = ['foo', 'fake-nodegroup1', 'fake-nodegroup2']
        verifylist = [('cluster', 'foo'), ('nodegroup', ['fake-nodegroup1', 'fake-nodegroup2'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.ng_mock.delete.assert_has_calls([call('foo', 'fake-nodegroup1'), call('foo', 'fake-nodegroup2')])

    def test_nodegroup_delete_no_args(self):
        arglist = []
        verifylist = [('cluster', ''), ('nodegroup', [])]
        self.assertRaises(magnum_fakes.MagnumParseException, self.check_parser, self.cmd, arglist, verifylist)