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
class TestClusterUpdate(TestCluster):

    def setUp(self):
        super(TestClusterUpdate, self).setUp()
        self.clusters_mock.update = mock.Mock()
        self.clusters_mock.update.return_value = None
        self.cmd = osc_clusters.UpdateCluster(self.app, None)

    def test_cluster_update_pass(self):
        arglist = ['foo', 'remove', 'bar']
        verifylist = [('cluster', 'foo'), ('op', 'remove'), ('attributes', [['bar']]), ('rollback', False)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.clusters_mock.update.assert_called_with('foo', [{'op': 'remove', 'path': '/bar'}])

    def test_cluster_update_bad_op(self):
        arglist = ['foo', 'bar', 'snafu']
        verifylist = [('cluster', 'foo'), ('op', 'bar'), ('attributes', ['snafu']), ('rollback', False)]
        self.assertRaises(magnum_fakes.MagnumParseException, self.check_parser, self.cmd, arglist, verifylist)