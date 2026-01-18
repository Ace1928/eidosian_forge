from unittest import mock
from magnumclient.osc.v1 import certificates as osc_certificates
from magnumclient.tests.osc.unit.v1 import fakes as magnum_fakes
class TestRotateCa(TestCertificate):

    def setUp(self):
        super(TestRotateCa, self).setUp()
        attr = dict()
        attr['name'] = 'fake-cluster-1'
        self._cluster = magnum_fakes.FakeCluster.create_one_cluster(attr)
        self.clusters_mock.get = mock.Mock()
        self.clusters_mock.get.return_value = self._cluster
        self.cmd = osc_certificates.RotateCa(self.app, None)

    def test_rotate_ca(self):
        arglist = ['fake-cluster']
        verifylist = [('cluster', 'fake-cluster')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        self.cmd.take_action(parsed_args)
        self.clusters_mock.get.assert_called_once_with('fake-cluster')

    def test_rotate_ca_missing_args(self):
        arglist = []
        verifylist = []
        self.assertRaises(magnum_fakes.MagnumParseException, self.check_parser, self.cmd, arglist, verifylist)