from unittest import mock
from neutronclient.osc.v2.dynamic_routing import bgp_peer
from neutronclient.tests.unit.osc.v2.dynamic_routing import fakes
class TestShowBgpPeer(fakes.TestNeutronDynamicRoutingOSCV2):
    _one_bgp_peer = fakes.FakeBgpPeer.create_one_bgp_peer()
    data = (_one_bgp_peer['auth_type'], _one_bgp_peer['id'], _one_bgp_peer['name'], _one_bgp_peer['peer_ip'], _one_bgp_peer['tenant_id'], _one_bgp_peer['remote_as'])
    _bgp_peer = _one_bgp_peer
    _bgp_peer_name = _one_bgp_peer['name']
    columns = ('auth_type', 'id', 'name', 'peer_ip', 'project_id', 'remote_as')

    def setUp(self):
        super(TestShowBgpPeer, self).setUp()
        self.networkclient.get_bgp_peer = mock.Mock(return_value=self._bgp_peer)
        self.cmd = bgp_peer.ShowBgpPeer(self.app, self.namespace)

    def test_bgp_peer_show(self):
        arglist = [self._bgp_peer_name]
        verifylist = [('bgp_peer', self._bgp_peer_name)]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        data = self.cmd.take_action(parsed_args)
        self.networkclient.get_bgp_peer.assert_called_once_with(self._bgp_peer_name)
        self.assertEqual(self.columns, data[0])
        self.assertEqual(self.data, data[1])