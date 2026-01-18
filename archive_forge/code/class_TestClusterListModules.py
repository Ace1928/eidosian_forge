from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from troveclient import common
from troveclient.osc.v1 import database_clusters
from troveclient.tests.osc.v1 import fakes
class TestClusterListModules(TestClusters):
    columns = database_clusters.ListDatabaseClusterModules.columns
    values = [('test-clstr-member-1', 'mymod1', 'ping', 'md5-1', '2018-04-17 05:34:02.84', '2018-04-17 05:34:02.84'), ('test-clstr-member-2', 'mymod2', 'ping', 'md5-2', '2018-04-17 05:34:02.84', '2018-04-17 05:34:02.84')]

    def setUp(self):
        super(TestClusterListModules, self).setUp()
        self.cmd = database_clusters.ListDatabaseClusterModules(self.app, None)
        self.data = self.fake_clusters.get_clusters_cls_1234()
        self.instance_client.modules.side_effect = self.fake_clusters.cluster_instance_modules()

    @mock.patch.object(utils, 'find_resource')
    def test_cluster_list_modules(self, mock_find_resource):
        mock_find_resource.return_value = self.data
        args = ['cls-1234']
        parsed_args = self.check_parser(self.cmd, args, [])
        columns, data = self.cmd.take_action(parsed_args)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.values, data)