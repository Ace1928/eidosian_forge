from unittest import mock
from neutronclient.osc.v2.dynamic_routing import bgp_peer
from neutronclient.tests.unit.osc.v2.dynamic_routing import fakes
class TestListBgpPeer(fakes.TestNeutronDynamicRoutingOSCV2):
    _bgp_peers = fakes.FakeBgpPeer.create_bgp_peers(count=1)
    columns = ('ID', 'Name', 'Peer IP', 'Remote AS')
    data = []
    for _bgp_peer in _bgp_peers:
        data.append((_bgp_peer['id'], _bgp_peer['name'], _bgp_peer['peer_ip'], _bgp_peer['remote_as']))

    def setUp(self):
        super(TestListBgpPeer, self).setUp()
        self.networkclient.bgp_peers = mock.Mock(return_value=self._bgp_peers)
        self.cmd = bgp_peer.ListBgpPeer(self.app, self.namespace)

    def test_bgp_peer_list(self):
        parsed_args = self.check_parser(self.cmd, [], [])
        columns, data = self.cmd.take_action(parsed_args)
        self.networkclient.bgp_peers.assert_called_once_with(retrieve_all=True)
        self.assertEqual(self.columns, columns)
        self.assertEqual(self.data, list(data))