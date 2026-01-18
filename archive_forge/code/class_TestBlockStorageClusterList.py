from cinderclient import api_versions
from osc_lib import exceptions
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import block_storage_cluster
class TestBlockStorageClusterList(TestBlockStorageCluster):
    fake_clusters = volume_fakes.create_clusters()

    def setUp(self):
        super().setUp()
        self.cluster_mock.list.return_value = self.fake_clusters
        self.cmd = block_storage_cluster.ListBlockStorageCluster(self.app, None)

    def test_cluster_list(self):
        self.volume_client.api_version = api_versions.APIVersion('3.7')
        arglist = []
        verifylist = [('cluster', None), ('binary', None), ('is_up', None), ('is_disabled', None), ('num_hosts', None), ('num_down_hosts', None), ('long', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        expected_columns = ('Name', 'Binary', 'State', 'Status')
        expected_data = tuple(((cluster.name, cluster.binary, cluster.state, cluster.status) for cluster in self.fake_clusters))
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(expected_columns, columns)
        self.assertEqual(expected_data, tuple(data))
        self.cluster_mock.list.assert_called_with(name=None, binary=None, is_up=None, disabled=None, num_hosts=None, num_down_hosts=None, detailed=False)

    def test_cluster_list_with_full_options(self):
        self.volume_client.api_version = api_versions.APIVersion('3.7')
        arglist = ['--cluster', 'foo', '--binary', 'bar', '--up', '--disabled', '--num-hosts', '5', '--num-down-hosts', '0', '--long']
        verifylist = [('cluster', 'foo'), ('binary', 'bar'), ('is_up', True), ('is_disabled', True), ('num_hosts', 5), ('num_down_hosts', 0), ('long', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        expected_columns = ('Name', 'Binary', 'State', 'Status', 'Num Hosts', 'Num Down Hosts', 'Last Heartbeat', 'Disabled Reason', 'Created At', 'Updated At')
        expected_data = tuple(((cluster.name, cluster.binary, cluster.state, cluster.status, cluster.num_hosts, cluster.num_down_hosts, cluster.last_heartbeat, cluster.disabled_reason, cluster.created_at, cluster.updated_at) for cluster in self.fake_clusters))
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(expected_columns, columns)
        self.assertEqual(expected_data, tuple(data))
        self.cluster_mock.list.assert_called_with(name='foo', binary='bar', is_up=True, disabled=True, num_hosts=5, num_down_hosts=0, detailed=True)

    def test_cluster_list_pre_v37(self):
        self.volume_client.api_version = api_versions.APIVersion('3.6')
        arglist = []
        verifylist = [('cluster', None), ('binary', None), ('is_up', None), ('is_disabled', None), ('num_hosts', None), ('num_down_hosts', None), ('long', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('--os-volume-api-version 3.7 or greater is required', str(exc))