from osc_lib.tests import utils as osc_utils
from unittest import mock
from saharaclient.api import cluster_templates as api_ct
from saharaclient.api import clusters as api_cl
from saharaclient.api import images as api_img
from saharaclient.api import node_group_templates as api_ngt
from saharaclient.osc.v2 import clusters as osc_cl
from saharaclient.tests.unit.osc.v1 import test_clusters as tc_v1
class TestUpdateCluster(TestClusters):

    def setUp(self):
        super(TestUpdateCluster, self).setUp()
        self.cl_mock.update.return_value = mock.Mock(cluster=CLUSTER_INFO.copy())
        self.cl_mock.find_unique.return_value = api_cl.Cluster(None, CLUSTER_INFO)
        self.cmd = osc_cl.UpdateCluster(self.app, None)

    def test_cluster_update_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(osc_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_cluster_update_nothing_updated(self):
        arglist = ['fake']
        verifylist = [('cluster', 'fake')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.cl_mock.update.assert_called_once_with('cluster_id')

    def test_cluster_update_all_options(self):
        arglist = ['fake', '--name', 'fake', '--description', 'descr', '--public', '--protected']
        verifylist = [('cluster', 'fake'), ('name', 'fake'), ('description', 'descr'), ('is_public', True), ('is_protected', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.cl_mock.update.assert_called_once_with('cluster_id', description='descr', is_protected=True, is_public=True, name='fake')
        expected_columns = ('Anti affinity', 'Cluster template id', 'Description', 'Id', 'Image', 'Is protected', 'Is public', 'Name', 'Neutron management network', 'Node groups', 'Plugin name', 'Plugin version', 'Status', 'Use autoconfig', 'User keypair id')
        self.assertEqual(expected_columns, columns)
        expected_data = ('', 'ct_id', 'Cluster template for tests', 'cluster_id', 'img_id', False, False, 'fake', 'net_id', 'fakeng:2', 'fake', '0.1', 'Active', True, 'test')
        self.assertEqual(expected_data, data)

    def test_cluster_update_private_unprotected(self):
        arglist = ['fake', '--private', '--unprotected']
        verifylist = [('cluster', 'fake'), ('is_public', False), ('is_protected', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.cl_mock.update.assert_called_once_with('cluster_id', is_protected=False, is_public=False)