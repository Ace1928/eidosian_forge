from unittest import mock
from magnumclient.osc.v1 import stats as osc_stats
from magnumclient.tests.osc.unit.v1 import fakes as magnum_fakes
class TestStatsList(TestStats):

    def setUp(self):
        super(TestStatsList, self).setUp()
        attr = dict()
        attr['name'] = 'fake-cluster-1'
        attr['project_id'] = 'abc'
        attr['node_count'] = 2
        attr['master_count'] = 1
        self._cluster = magnum_fakes.FakeCluster.create_one_cluster(attr)
        self.clusters_mock.list = mock.Mock()
        self.clusters_mock.list.return_value = self._cluster
        self.cmd = osc_stats.ListStats(self.app, None)

    def test_stats_list(self):
        arglist = ['abc']
        verifylist = [('project_id', 'abc')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.clusters_mock.list.assert_called_once_with(project_id='abc')

    def test_stats_list_wrong_projectid(self):
        arglist = ['abcd']
        verifylist = [('project_id', 'abcd')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.clusters_mock.list.assert_called_once_with(project_id='abcd')

    def test_stats_list_missing_args(self):
        arglist = []
        verifylist = []
        self.assertRaises(magnum_fakes.MagnumParseException, self.check_parser, self.cmd, arglist, verifylist)