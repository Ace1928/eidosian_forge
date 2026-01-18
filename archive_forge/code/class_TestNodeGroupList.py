import copy
from unittest import mock
from unittest.mock import call
from magnumclient.osc.v1 import nodegroups as osc_nodegroups
from magnumclient.tests.osc.unit.v1 import fakes as magnum_fakes
class TestNodeGroupList(TestNodeGroup):
    nodegroup = magnum_fakes.FakeNodeGroup.create_one_nodegroup()
    columns = ['uuid', 'name', 'flavor_id', 'image_id', 'node_count', 'status', 'role']
    datalist = ((nodegroup.uuid, nodegroup.name, nodegroup.flavor_id, nodegroup.image_id, nodegroup.node_count, nodegroup.status, nodegroup.role),)

    def setUp(self):
        super(TestNodeGroupList, self).setUp()
        self.ng_mock.list = mock.Mock()
        self.ng_mock.list.return_value = [self.nodegroup]
        self.cmd = osc_nodegroups.ListNodeGroup(self.app, None)

    def test_nodegroup_list_no_options(self):
        arglist = []
        verifylist = [('cluster', ''), ('limit', None), ('sort_key', None), ('sort_dir', None)]
        self.assertRaises(magnum_fakes.MagnumParseException, self.check_parser, self.cmd, arglist, verifylist)

    def test_nodegroup_list_ok(self):
        arglist = ['fake-cluster']
        verifylist = [('cluster', 'fake-cluster'), ('limit', None), ('sort_key', None), ('sort_dir', None)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.ng_mock.list.assert_called_with('fake-cluster', limit=None, sort_dir=None, sort_key=None, role=None)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.datalist, tuple(data))

    def test_nodegroup_list_options(self):
        arglist = ['fake-cluster', '--limit', '1', '--sort-key', 'key', '--sort-dir', 'asc']
        verifylist = [('cluster', 'fake-cluster'), ('limit', 1), ('sort_key', 'key'), ('sort_dir', 'asc')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.ng_mock.list.assert_called_with('fake-cluster', limit=1, sort_dir='asc', sort_key='key', role=None)