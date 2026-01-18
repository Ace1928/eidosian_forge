from unittest import mock
from neutronclient.osc.v2.dynamic_routing import bgp_peer
from neutronclient.tests.unit.osc.v2.dynamic_routing import fakes
class TestSetBgpPeer(fakes.TestNeutronDynamicRoutingOSCV2):
    _one_bgp_peer = fakes.FakeBgpPeer.create_one_bgp_peer()
    _bgp_peer_name = _one_bgp_peer['name']

    def setUp(self):
        super(TestSetBgpPeer, self).setUp()
        self.networkclient.update_bgp_peer = mock.Mock(return_value=None)
        bgp_peer.get_bgp_peer_id = mock.Mock(return_value=self._bgp_peer_name)
        self.cmd = bgp_peer.SetBgpPeer(self.app, self.namespace)

    def test_set_bgp_peer(self):
        arglist = [self._bgp_peer_name, '--name', 'noob']
        verifylist = [('bgp_peer', self._bgp_peer_name), ('name', 'noob')]
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        attrs = {'name': 'noob', 'password': None}
        self.networkclient.update_bgp_peer.assert_called_once_with(self._bgp_peer_name, **attrs)
        self.assertIsNone(result)