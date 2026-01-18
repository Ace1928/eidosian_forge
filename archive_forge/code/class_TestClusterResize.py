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
class TestClusterResize(TestCluster):

    def setUp(self):
        super(TestClusterResize, self).setUp()
        self.cluster = mock.Mock()
        self.cluster.uuid = 'UUID1'
        self.clusters_mock.resize = mock.Mock()
        self.clusters_mock.resize.return_value = None
        self.clusters_mock.get = mock.Mock()
        self.clusters_mock.get.return_value = self.cluster
        self.cmd = osc_clusters.ResizeCluster(self.app, None)

    def test_cluster_resize_pass(self):
        arglist = ['foo', '2']
        verifylist = [('cluster', 'foo'), ('node_count', 2), ('nodes_to_remove', None), ('nodegroup', None)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.clusters_mock.resize.assert_called_with('UUID1', 2, None, None)

    def test_cluster_resize_to_zero_pass(self):
        arglist = ['foo', '0']
        verifylist = [('cluster', 'foo'), ('node_count', 0), ('nodes_to_remove', None), ('nodegroup', None)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.clusters_mock.resize.assert_called_with('UUID1', 0, None, None)