import copy
from unittest import mock
from unittest.mock import call
from magnumclient.osc.v1 import nodegroups as osc_nodegroups
from magnumclient.tests.osc.unit.v1 import fakes as magnum_fakes
class TestNodeGroupCreate(TestNodeGroup):

    def setUp(self):
        super(TestNodeGroupCreate, self).setUp()
        self.nodegroup = magnum_fakes.FakeNodeGroup.create_one_nodegroup()
        self.ng_mock.create = mock.Mock()
        self.ng_mock.create.return_value = self.nodegroup
        self.ng_mock.get = mock.Mock()
        self.ng_mock.get.return_value = copy.deepcopy(self.nodegroup)
        self.ng_mock.update = mock.Mock()
        self.ng_mock.update.return_value = self.nodegroup
        self._default_args = {'name': 'fake-nodegroup', 'node_count': 1, 'role': 'worker', 'min_node_count': 0, 'max_node_count': None}
        self.cmd = osc_nodegroups.CreateNodeGroup(self.app, None)
        self.data = tuple(map(lambda x: getattr(self.nodegroup, x), osc_nodegroups.NODEGROUP_ATTRIBUTES))

    def test_nodegroup_create_required_args_pass(self):
        """Verifies required arguments."""
        arglist = [self.nodegroup.cluster_id, self.nodegroup.name]
        verifylist = [('cluster', self.nodegroup.cluster_id), ('name', self.nodegroup.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.ng_mock.create.assert_called_with(self.nodegroup.cluster_id, **self._default_args)

    def test_nodegroup_create_missing_required_arg(self):
        """Verifies missing required arguments."""
        arglist = [self.nodegroup.name]
        verifylist = [('name', self.nodegroup.name)]
        self.assertRaises(magnum_fakes.MagnumParseException, self.check_parser, self.cmd, arglist, verifylist)

    def test_nodegroup_create_with_labels(self):
        """Verifies labels are properly parsed when given as argument."""
        expected_args = self._default_args
        expected_args['labels'] = {'arg1': 'value1', 'arg2': 'value2'}
        arglist = ['--labels', 'arg1=value1', '--labels', 'arg2=value2', self.nodegroup.cluster_id, self.nodegroup.name]
        verifylist = [('labels', ['arg1=value1', 'arg2=value2']), ('name', self.nodegroup.name), ('cluster', self.nodegroup.cluster_id)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.ng_mock.create.assert_called_with(self.nodegroup.cluster_id, **expected_args)