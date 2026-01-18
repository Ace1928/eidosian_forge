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
class TestClusterDelete(TestCluster):

    def setUp(self):
        super(TestClusterDelete, self).setUp()
        self.clusters_mock.delete = mock.Mock()
        self.clusters_mock.delete.return_value = None
        self.cmd = osc_clusters.DeleteCluster(self.app, None)

    def test_cluster_delete_one(self):
        arglist = ['foo']
        verifylist = [('cluster', ['foo'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.clusters_mock.delete.assert_called_with('foo')

    def test_cluster_delete_multiple(self):
        arglist = ['foo', 'bar']
        verifylist = [('cluster', ['foo', 'bar'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.clusters_mock.delete.assert_has_calls([call('foo'), call('bar')])

    def test_cluster_delete_bad_uuid(self):
        arglist = ['foo']
        verifylist = [('cluster', ['foo'])]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        returns = self.cmd.take_action(parsed_args)
        self.assertEqual(returns, None)

    def test_cluster_delete_no_uuid(self):
        arglist = []
        verifylist = [('cluster', [])]
        self.assertRaises(magnum_fakes.MagnumParseException, self.check_parser, self.cmd, arglist, verifylist)