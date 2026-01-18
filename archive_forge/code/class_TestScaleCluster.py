from osc_lib.tests import utils as osc_utils
from unittest import mock
from saharaclient.api import cluster_templates as api_ct
from saharaclient.api import clusters as api_cl
from saharaclient.api import images as api_img
from saharaclient.api import node_group_templates as api_ngt
from saharaclient.osc.v2 import clusters as osc_cl
from saharaclient.tests.unit.osc.v1 import test_clusters as tc_v1
class TestScaleCluster(TestClusters):

    def setUp(self):
        super(TestScaleCluster, self).setUp()
        self.cl_mock.scale.return_value = mock.Mock(cluster=CLUSTER_INFO.copy())
        self.cl_mock.find_unique.return_value = api_cl.Cluster(None, CLUSTER_INFO)
        self.cmd = osc_cl.ScaleCluster(self.app, None)

    def test_cluster_scale_no_options(self):
        arglist = []
        verifylist = []
        self.assertRaises(osc_utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)

    def test_cluster_scale_resize(self):
        self.ngt_mock.find_unique.return_value = api_ngt.NodeGroupTemplate(None, NGT_INFO)
        arglist = ['fake', '--instances', 'fakeng:1']
        verifylist = [('cluster', 'fake'), ('instances', ['fakeng:1'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.cl_mock.scale.assert_called_once_with('cluster_id', {'resize_node_groups': [{'count': 1, 'name': 'fakeng'}]})
        expected_columns = ('Anti affinity', 'Cluster template id', 'Description', 'Id', 'Image', 'Is protected', 'Is public', 'Name', 'Neutron management network', 'Node groups', 'Plugin name', 'Plugin version', 'Status', 'Use autoconfig', 'User keypair id')
        self.assertEqual(expected_columns, columns)
        expected_data = ('', 'ct_id', 'Cluster template for tests', 'cluster_id', 'img_id', False, False, 'fake', 'net_id', 'fakeng:2', 'fake', '0.1', 'Active', True, 'test')
        self.assertEqual(expected_data, data)

    def test_cluster_scale_add_ng(self):
        new_ng = {'name': 'new', 'id': 'new_id'}
        self.ngt_mock.find_unique.return_value = api_ngt.NodeGroupTemplate(None, new_ng)
        arglist = ['fake', '--instances', 'new:1']
        verifylist = [('cluster', 'fake'), ('instances', ['new:1'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.cl_mock.scale.assert_called_once_with('cluster_id', {'add_node_groups': [{'count': 1, 'node_group_template_id': 'new_id', 'name': 'new'}]})