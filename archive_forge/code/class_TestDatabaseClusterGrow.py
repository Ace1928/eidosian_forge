from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from troveclient import common
from troveclient.osc.v1 import database_clusters
from troveclient.tests.osc.v1 import fakes
class TestDatabaseClusterGrow(TestClusters):

    def setUp(self):
        super(TestDatabaseClusterGrow, self).setUp()
        self.cmd = database_clusters.GrowDatabaseCluster(self.app, None)

    @mock.patch.object(utils, 'find_resource')
    @mock.patch.object(database_clusters, '_parse_instance_options')
    def test_cluster_grow(self, mock_parse_instance_opts, mock_find_resource):
        args = ['test-clstr', '--instance', 'name=test-clstr-member-3,flavor=3']
        parsed_instance_opts = [{'name': 'test-clstr-member-3', 'flavor': 3}]
        mock_parse_instance_opts.return_value = parsed_instance_opts
        mock_find_resource.return_value = args[0]
        parsed_args = self.check_parser(self.cmd, args, [])
        result = self.cmd.take_action(parsed_args)
        self.cluster_client.grow.assert_called_with('test-clstr', instances=parsed_instance_opts)
        self.assertIsNone(result)