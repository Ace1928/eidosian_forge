from osc_lib.tests import utils as osc_utils
from unittest import mock
from saharaclient.api import cluster_templates as api_ct
from saharaclient.api import clusters as api_cl
from saharaclient.api import images as api_img
from saharaclient.api import node_group_templates as api_ngt
from saharaclient.osc.v2 import clusters as osc_cl
from saharaclient.tests.unit.osc.v1 import test_clusters as tc_v1
class TestDeleteCluster(TestClusters):

    def setUp(self):
        super(TestDeleteCluster, self).setUp()
        self.cl_mock.find_unique.return_value = api_cl.Cluster(None, CLUSTER_INFO)
        self.cmd = osc_cl.DeleteCluster(self.app, None)

    def test_cluster_delete(self):
        arglist = ['fake']
        verifylist = [('cluster', ['fake'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.cl_mock.delete.assert_called_once_with('cluster_id')