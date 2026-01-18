from osc_lib.tests import utils as osc_utils
from unittest import mock
from saharaclient.api import cluster_templates as api_ct
from saharaclient.api import clusters as api_cl
from saharaclient.api import images as api_img
from saharaclient.api import node_group_templates as api_ngt
from saharaclient.osc.v2 import clusters as osc_cl
from saharaclient.tests.unit.osc.v1 import test_clusters as tc_v1
class TestVerificationUpdateCluster(TestClusters):

    def setUp(self):
        super(TestVerificationUpdateCluster, self).setUp()
        self.cl_mock.find_unique.return_value = api_cl.Cluster(None, CLUSTER_INFO)
        self.cl_mock.verification_update.return_value = api_cl.Cluster(None, CLUSTER_INFO)
        self.cmd = osc_cl.VerificationUpdateCluster(self.app, None)

    def test_verification_show(self):
        arglist = ['fake', '--show']
        verifylist = [('cluster', 'fake'), ('show', True)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        columns, data = self.cmd.take_action(parsed_args)
        self.cl_mock.find_unique.assert_called_once_with(name='fake')
        expected_columns = ('Health check (some check)', 'Verification status')
        self.assertEqual(expected_columns, columns)
        expected_data = ('GREEN', 'GREEN')
        self.assertEqual(expected_data, data)

    def test_verification_start(self):
        arglist = ['fake', '--start']
        verifylist = [('cluster', 'fake'), ('status', 'START')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.cl_mock.verification_update.assert_called_once_with('cluster_id', 'START')