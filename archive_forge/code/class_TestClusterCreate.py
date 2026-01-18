import copy
import os
import sys
import tempfile
from unittest import mock
from contextlib import contextmanager
from unittest.mock import call
from magnumclient import exceptions
from magnumclient.osc.v1 import clusters as osc_clusters
from magnumclient.tests.osc.unit.v1 import fakes as magnum_fakes
class TestClusterCreate(TestCluster):

    def setUp(self):
        super(TestClusterCreate, self).setUp()
        attr = dict()
        attr['name'] = 'fake-cluster-1'
        self._cluster = magnum_fakes.FakeCluster.create_one_cluster(attr)
        self._default_args = {'cluster_template_id': 'fake-ct', 'create_timeout': 60, 'discovery_url': None, 'keypair': None, 'master_count': 1, 'name': 'fake-cluster-1', 'node_count': 1}
        self.clusters_mock.create = mock.Mock()
        self.clusters_mock.create.return_value = self._cluster
        self.clusters_mock.get = mock.Mock()
        self.clusters_mock.get.return_value = copy.deepcopy(self._cluster)
        self.clusters_mock.update = mock.Mock()
        self.clusters_mock.update.return_value = self._cluster
        self.cmd = osc_clusters.CreateCluster(self.app, None)
        self.data = tuple(map(lambda x: getattr(self._cluster, x), osc_clusters.CLUSTER_ATTRIBUTES))

    def test_cluster_create_required_args_pass(self):
        """Verifies required arguments."""
        arglist = ['--cluster-template', self._cluster.cluster_template_id, self._cluster.name]
        verifylist = [('cluster_template', self._cluster.cluster_template_id), ('name', self._cluster.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.clusters_mock.create.assert_called_with(**self._default_args)

    def test_cluster_create_missing_required_arg(self):
        """Verifies missing required arguments."""
        arglist = [self._cluster.name]
        verifylist = [('name', self._cluster.name)]
        self.assertRaises(magnum_fakes.MagnumParseException, self.check_parser, self.cmd, arglist, verifylist)

    def test_cluster_create_with_labels(self):
        """Verifies labels are properly parsed when given as argument."""
        expected_args = self._default_args
        expected_args['labels'] = {'arg1': 'value1', 'arg2': 'value2'}
        arglist = ['--cluster-template', self._cluster.cluster_template_id, '--labels', 'arg1=value1', '--labels', 'arg2=value2', self._cluster.name]
        verifylist = [('cluster_template', self._cluster.cluster_template_id), ('labels', ['arg1=value1', 'arg2=value2']), ('name', self._cluster.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.clusters_mock.create.assert_called_with(**expected_args)

    def test_cluster_create_with_lb_disabled(self):
        """Verifies master lb disabled properly parsed."""
        expected_args = self._default_args
        expected_args['master_lb_enabled'] = False
        arglist = ['--cluster-template', self._cluster.cluster_template_id, '--master-lb-disabled', self._cluster.name]
        verifylist = [('cluster_template', self._cluster.cluster_template_id), ('master_lb_enabled', [False]), ('name', self._cluster.name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.clusters_mock.create.assert_called_with(**expected_args)