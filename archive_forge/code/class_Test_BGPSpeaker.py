import logging
import unittest
from unittest import mock
from os_ken.services.protocols.bgp import bgpspeaker
from os_ken.services.protocols.bgp.bgpspeaker import EVPN_MAX_ET
from os_ken.services.protocols.bgp.bgpspeaker import ESI_TYPE_LACP
from os_ken.services.protocols.bgp.api.prefix import ESI_TYPE_L2_BRIDGE
from os_ken.services.protocols.bgp.bgpspeaker import ESI_TYPE_MAC_BASED
from os_ken.services.protocols.bgp.api.prefix import REDUNDANCY_MODE_ALL_ACTIVE
from os_ken.services.protocols.bgp.api.prefix import REDUNDANCY_MODE_SINGLE_ACTIVE
class Test_BGPSpeaker(unittest.TestCase):
    """
    Test case for bgp.bgpspeaker.BGPSpeaker
    """

    @mock.patch('os_ken.services.protocols.bgp.bgpspeaker.BGPSpeaker.__init__', mock.MagicMock(return_value=None))
    @mock.patch('os_ken.services.protocols.bgp.bgpspeaker.call')
    def test_evpn_prefix_add_eth_auto_discovery(self, mock_call):
        route_type = bgpspeaker.EVPN_ETH_AUTO_DISCOVERY
        route_dist = '65000:100'
        esi = {'type': ESI_TYPE_LACP, 'mac_addr': 'aa:bb:cc:dd:ee:ff', 'port_key': 100}
        ethernet_tag_id = EVPN_MAX_ET
        redundancy_mode = REDUNDANCY_MODE_ALL_ACTIVE
        next_hop = '0.0.0.0'
        expected_kwargs = {'route_type': route_type, 'route_dist': route_dist, 'esi': esi, 'ethernet_tag_id': ethernet_tag_id, 'redundancy_mode': redundancy_mode, 'next_hop': next_hop}
        speaker = bgpspeaker.BGPSpeaker(65000, '10.0.0.1')
        speaker.evpn_prefix_add(route_type=route_type, route_dist=route_dist, esi=esi, ethernet_tag_id=ethernet_tag_id, redundancy_mode=redundancy_mode)
        mock_call.assert_called_with('evpn_prefix.add_local', **expected_kwargs)

    @mock.patch('os_ken.services.protocols.bgp.bgpspeaker.BGPSpeaker.__init__', mock.MagicMock(return_value=None))
    @mock.patch('os_ken.services.protocols.bgp.bgpspeaker.call')
    def test_evpn_prefix_add_eth_auto_discovery_vni(self, mock_call):
        route_type = bgpspeaker.EVPN_ETH_AUTO_DISCOVERY
        route_dist = '65000:100'
        esi = {'type': ESI_TYPE_L2_BRIDGE, 'mac_addr': 'aa:bb:cc:dd:ee:ff', 'priority': 100}
        ethernet_tag_id = EVPN_MAX_ET
        redundancy_mode = REDUNDANCY_MODE_SINGLE_ACTIVE
        vni = 500
        next_hop = '0.0.0.0'
        expected_kwargs = {'route_type': route_type, 'route_dist': route_dist, 'esi': esi, 'ethernet_tag_id': ethernet_tag_id, 'redundancy_mode': redundancy_mode, 'vni': vni, 'next_hop': next_hop}
        speaker = bgpspeaker.BGPSpeaker(65000, '10.0.0.1')
        speaker.evpn_prefix_add(route_type=route_type, route_dist=route_dist, esi=esi, ethernet_tag_id=ethernet_tag_id, redundancy_mode=redundancy_mode, vni=vni)
        mock_call.assert_called_with('evpn_prefix.add_local', **expected_kwargs)

    @mock.patch('os_ken.services.protocols.bgp.bgpspeaker.BGPSpeaker.__init__', mock.MagicMock(return_value=None))
    @mock.patch('os_ken.services.protocols.bgp.bgpspeaker.call')
    def test_evpn_prefix_add_mac_ip_adv(self, mock_call):
        route_type = bgpspeaker.EVPN_MAC_IP_ADV_ROUTE
        route_dist = '65000:100'
        esi = 0
        ethernet_tag_id = 200
        mac_addr = 'aa:bb:cc:dd:ee:ff'
        ip_addr = '192.168.0.1'
        next_hop = '10.0.0.1'
        expected_kwargs = {'route_type': route_type, 'route_dist': route_dist, 'esi': esi, 'ethernet_tag_id': ethernet_tag_id, 'mac_addr': mac_addr, 'ip_addr': ip_addr, 'next_hop': next_hop}
        speaker = bgpspeaker.BGPSpeaker(65000, '10.0.0.1')
        speaker.evpn_prefix_add(route_type=route_type, route_dist=route_dist, esi=esi, ethernet_tag_id=ethernet_tag_id, mac_addr=mac_addr, ip_addr=ip_addr, next_hop=next_hop)
        mock_call.assert_called_with('evpn_prefix.add_local', **expected_kwargs)

    @mock.patch('os_ken.services.protocols.bgp.bgpspeaker.BGPSpeaker.__init__', mock.MagicMock(return_value=None))
    @mock.patch('os_ken.services.protocols.bgp.bgpspeaker.call')
    def test_evpn_prefix_add_mac_ip_adv_vni(self, mock_call):
        route_type = bgpspeaker.EVPN_MAC_IP_ADV_ROUTE
        route_dist = '65000:100'
        esi = 0
        ethernet_tag_id = 200
        mac_addr = 'aa:bb:cc:dd:ee:ff'
        ip_addr = '192.168.0.1'
        vni = 500
        next_hop = '10.0.0.1'
        tunnel_type = bgpspeaker.TUNNEL_TYPE_VXLAN
        expected_kwargs = {'route_type': route_type, 'route_dist': route_dist, 'esi': esi, 'ethernet_tag_id': ethernet_tag_id, 'mac_addr': mac_addr, 'ip_addr': ip_addr, 'vni': vni, 'next_hop': next_hop, 'tunnel_type': tunnel_type}
        speaker = bgpspeaker.BGPSpeaker(65000, '10.0.0.1')
        speaker.evpn_prefix_add(route_type=route_type, route_dist=route_dist, esi=esi, ethernet_tag_id=ethernet_tag_id, mac_addr=mac_addr, ip_addr=ip_addr, vni=vni, next_hop=next_hop, tunnel_type=tunnel_type)
        mock_call.assert_called_with('evpn_prefix.add_local', **expected_kwargs)

    @mock.patch('os_ken.services.protocols.bgp.bgpspeaker.BGPSpeaker.__init__', mock.MagicMock(return_value=None))
    @mock.patch('os_ken.services.protocols.bgp.bgpspeaker.call')
    def test_evpn_prefix_add_multicast_etag(self, mock_call):
        route_type = bgpspeaker.EVPN_MULTICAST_ETAG_ROUTE
        route_dist = '65000:100'
        esi = 0
        ethernet_tag_id = 200
        mac_addr = 'aa:bb:cc:dd:ee:ff'
        ip_addr = '192.168.0.1'
        next_hop = '10.0.0.1'
        expected_kwargs = {'route_type': route_type, 'route_dist': route_dist, 'ethernet_tag_id': ethernet_tag_id, 'ip_addr': ip_addr, 'next_hop': next_hop}
        speaker = bgpspeaker.BGPSpeaker(65000, '10.0.0.1')
        speaker.evpn_prefix_add(route_type=route_type, route_dist=route_dist, esi=esi, ethernet_tag_id=ethernet_tag_id, mac_addr=mac_addr, ip_addr=ip_addr, next_hop=next_hop)
        mock_call.assert_called_with('evpn_prefix.add_local', **expected_kwargs)

    @mock.patch('os_ken.services.protocols.bgp.bgpspeaker.BGPSpeaker.__init__', mock.MagicMock(return_value=None))
    @mock.patch('os_ken.services.protocols.bgp.bgpspeaker.call')
    def test_evpn_prefix_add_multicast_etag_no_next_hop(self, mock_call):
        route_type = bgpspeaker.EVPN_MULTICAST_ETAG_ROUTE
        route_dist = '65000:100'
        esi = 0
        ethernet_tag_id = 200
        mac_addr = 'aa:bb:cc:dd:ee:ff'
        ip_addr = '192.168.0.1'
        next_hop = '0.0.0.0'
        expected_kwargs = {'route_type': route_type, 'route_dist': route_dist, 'ethernet_tag_id': ethernet_tag_id, 'ip_addr': ip_addr, 'next_hop': next_hop}
        speaker = bgpspeaker.BGPSpeaker(65000, '10.0.0.1')
        speaker.evpn_prefix_add(route_type=route_type, route_dist=route_dist, esi=esi, ethernet_tag_id=ethernet_tag_id, mac_addr=mac_addr, ip_addr=ip_addr)
        mock_call.assert_called_with('evpn_prefix.add_local', **expected_kwargs)

    @mock.patch('os_ken.services.protocols.bgp.bgpspeaker.BGPSpeaker.__init__', mock.MagicMock(return_value=None))
    @mock.patch('os_ken.services.protocols.bgp.bgpspeaker.call')
    def test_evpn_prefix_add_eth_segment(self, mock_call):
        route_type = bgpspeaker.EVPN_ETH_SEGMENT
        route_dist = '65000:100'
        esi = {'type': ESI_TYPE_MAC_BASED, 'mac_addr': 'aa:bb:cc:dd:ee:ff', 'local_disc': 100}
        ip_addr = '192.168.0.1'
        next_hop = '0.0.0.0'
        expected_kwargs = {'route_type': route_type, 'route_dist': route_dist, 'esi': esi, 'ip_addr': ip_addr, 'next_hop': next_hop}
        speaker = bgpspeaker.BGPSpeaker(65000, '10.0.0.1')
        speaker.evpn_prefix_add(route_type=route_type, route_dist=route_dist, esi=esi, ip_addr=ip_addr)
        mock_call.assert_called_with('evpn_prefix.add_local', **expected_kwargs)

    @mock.patch('os_ken.services.protocols.bgp.bgpspeaker.BGPSpeaker.__init__', mock.MagicMock(return_value=None))
    @mock.patch('os_ken.services.protocols.bgp.bgpspeaker.call')
    def test_evpn_prefix_add_ip_prefix_route(self, mock_call):
        route_type = bgpspeaker.EVPN_IP_PREFIX_ROUTE
        route_dist = '65000:100'
        esi = 0
        ethernet_tag_id = 200
        ip_prefix = '192.168.0.0/24'
        gw_ip_addr = '172.16.0.1'
        next_hop = '0.0.0.0'
        expected_kwargs = {'route_type': route_type, 'route_dist': route_dist, 'esi': esi, 'ethernet_tag_id': ethernet_tag_id, 'ip_prefix': ip_prefix, 'gw_ip_addr': gw_ip_addr, 'next_hop': next_hop}
        speaker = bgpspeaker.BGPSpeaker(65000, '10.0.0.1')
        speaker.evpn_prefix_add(route_type=route_type, route_dist=route_dist, esi=esi, ethernet_tag_id=ethernet_tag_id, ip_prefix=ip_prefix, gw_ip_addr=gw_ip_addr)
        mock_call.assert_called_with('evpn_prefix.add_local', **expected_kwargs)

    @mock.patch('os_ken.services.protocols.bgp.bgpspeaker.BGPSpeaker.__init__', mock.MagicMock(return_value=None))
    @mock.patch('os_ken.services.protocols.bgp.bgpspeaker.call')
    def test_evpn_prefix_add_ip_prefix_route_vni(self, mock_call):
        route_type = bgpspeaker.EVPN_IP_PREFIX_ROUTE
        route_dist = '65000:100'
        esi = 0
        ethernet_tag_id = 200
        ip_prefix = '192.168.0.0/24'
        gw_ip_addr = '172.16.0.1'
        vni = 500
        tunnel_type = bgpspeaker.TUNNEL_TYPE_VXLAN
        next_hop = '0.0.0.0'
        expected_kwargs = {'route_type': route_type, 'route_dist': route_dist, 'esi': esi, 'ethernet_tag_id': ethernet_tag_id, 'ip_prefix': ip_prefix, 'gw_ip_addr': gw_ip_addr, 'tunnel_type': tunnel_type, 'vni': vni, 'next_hop': next_hop}
        speaker = bgpspeaker.BGPSpeaker(65000, '10.0.0.1')
        speaker.evpn_prefix_add(route_type=route_type, route_dist=route_dist, esi=esi, ethernet_tag_id=ethernet_tag_id, ip_prefix=ip_prefix, gw_ip_addr=gw_ip_addr, tunnel_type=tunnel_type, vni=vni)
        mock_call.assert_called_with('evpn_prefix.add_local', **expected_kwargs)

    @mock.patch('os_ken.services.protocols.bgp.bgpspeaker.BGPSpeaker.__init__', mock.MagicMock(return_value=None))
    @mock.patch('os_ken.services.protocols.bgp.bgpspeaker.call')
    def test_evpn_prefix_add_invalid_route_type(self, mock_call):
        route_type = 'foobar'
        route_dist = '65000:100'
        esi = 0
        ethernet_tag_id = 200
        mac_addr = 'aa:bb:cc:dd:ee:ff'
        ip_addr = '192.168.0.1'
        next_hop = '10.0.0.1'
        speaker = bgpspeaker.BGPSpeaker(65000, '10.0.0.1')
        self.assertRaises(ValueError, speaker.evpn_prefix_add, route_type=route_type, route_dist=route_dist, esi=esi, ethernet_tag_id=ethernet_tag_id, mac_addr=mac_addr, ip_addr=ip_addr, next_hop=next_hop)

    @mock.patch('os_ken.services.protocols.bgp.bgpspeaker.BGPSpeaker.__init__', mock.MagicMock(return_value=None))
    @mock.patch('os_ken.services.protocols.bgp.bgpspeaker.call')
    def test_evpn_prefix_del_auto_discovery(self, mock_call):
        route_type = bgpspeaker.EVPN_ETH_AUTO_DISCOVERY
        route_dist = '65000:100'
        esi = {'type': ESI_TYPE_LACP, 'mac_addr': 'aa:bb:cc:dd:ee:ff', 'port_key': 100}
        ethernet_tag_id = EVPN_MAX_ET
        expected_kwargs = {'route_type': route_type, 'route_dist': route_dist, 'esi': esi, 'ethernet_tag_id': ethernet_tag_id}
        speaker = bgpspeaker.BGPSpeaker(65000, '10.0.0.1')
        speaker.evpn_prefix_del(route_type=route_type, route_dist=route_dist, esi=esi, ethernet_tag_id=ethernet_tag_id)
        mock_call.assert_called_with('evpn_prefix.delete_local', **expected_kwargs)

    @mock.patch('os_ken.services.protocols.bgp.bgpspeaker.BGPSpeaker.__init__', mock.MagicMock(return_value=None))
    @mock.patch('os_ken.services.protocols.bgp.bgpspeaker.call')
    def test_evpn_prefix_del_mac_ip_adv(self, mock_call):
        route_type = bgpspeaker.EVPN_MAC_IP_ADV_ROUTE
        route_dist = '65000:100'
        ethernet_tag_id = 200
        mac_addr = 'aa:bb:cc:dd:ee:ff'
        ip_addr = '192.168.0.1'
        expected_kwargs = {'route_type': route_type, 'route_dist': route_dist, 'ethernet_tag_id': ethernet_tag_id, 'mac_addr': mac_addr, 'ip_addr': ip_addr}
        speaker = bgpspeaker.BGPSpeaker(65000, '10.0.0.1')
        speaker.evpn_prefix_del(route_type=route_type, route_dist=route_dist, ethernet_tag_id=ethernet_tag_id, mac_addr=mac_addr, ip_addr=ip_addr)
        mock_call.assert_called_with('evpn_prefix.delete_local', **expected_kwargs)

    @mock.patch('os_ken.services.protocols.bgp.bgpspeaker.BGPSpeaker.__init__', mock.MagicMock(return_value=None))
    @mock.patch('os_ken.services.protocols.bgp.bgpspeaker.call')
    def test_evpn_prefix_del_multicast_etag(self, mock_call):
        route_type = bgpspeaker.EVPN_MULTICAST_ETAG_ROUTE
        route_dist = '65000:100'
        esi = 0
        ethernet_tag_id = 200
        mac_addr = 'aa:bb:cc:dd:ee:ff'
        ip_addr = '192.168.0.1'
        expected_kwargs = {'route_type': route_type, 'route_dist': route_dist, 'ethernet_tag_id': ethernet_tag_id, 'ip_addr': ip_addr}
        speaker = bgpspeaker.BGPSpeaker(65000, '10.0.0.1')
        speaker.evpn_prefix_del(route_type=route_type, route_dist=route_dist, esi=esi, ethernet_tag_id=ethernet_tag_id, mac_addr=mac_addr, ip_addr=ip_addr)
        mock_call.assert_called_with('evpn_prefix.delete_local', **expected_kwargs)

    @mock.patch('os_ken.services.protocols.bgp.bgpspeaker.BGPSpeaker.__init__', mock.MagicMock(return_value=None))
    @mock.patch('os_ken.services.protocols.bgp.bgpspeaker.call')
    def test_evpn_prefix_del_invalid_route_type(self, mock_call):
        route_type = 'foobar'
        route_dist = '65000:100'
        esi = 0
        ethernet_tag_id = 200
        mac_addr = 'aa:bb:cc:dd:ee:ff'
        ip_addr = '192.168.0.1'
        speaker = bgpspeaker.BGPSpeaker(65000, '10.0.0.1')
        self.assertRaises(ValueError, speaker.evpn_prefix_del, route_type=route_type, route_dist=route_dist, esi=esi, ethernet_tag_id=ethernet_tag_id, mac_addr=mac_addr, ip_addr=ip_addr)

    @mock.patch('os_ken.services.protocols.bgp.bgpspeaker.BGPSpeaker.__init__', mock.MagicMock(return_value=None))
    @mock.patch('os_ken.services.protocols.bgp.bgpspeaker.call')
    def test_evpn_prefix_del_eth_segment(self, mock_call):
        route_type = bgpspeaker.EVPN_ETH_SEGMENT
        route_dist = '65000:100'
        esi = {'esi_type': ESI_TYPE_MAC_BASED, 'mac_addr': 'aa:bb:cc:dd:ee:ff', 'local_disc': 100}
        ip_addr = '192.168.0.1'
        expected_kwargs = {'route_type': route_type, 'route_dist': route_dist, 'esi': esi, 'ip_addr': ip_addr}
        speaker = bgpspeaker.BGPSpeaker(65000, '10.0.0.1')
        speaker.evpn_prefix_del(route_type=route_type, route_dist=route_dist, esi=esi, ip_addr=ip_addr)
        mock_call.assert_called_with('evpn_prefix.delete_local', **expected_kwargs)

    @mock.patch('os_ken.services.protocols.bgp.bgpspeaker.BGPSpeaker.__init__', mock.MagicMock(return_value=None))
    @mock.patch('os_ken.services.protocols.bgp.bgpspeaker.call')
    def test_evpn_prefix_del_ip_prefix_route(self, mock_call):
        route_type = bgpspeaker.EVPN_IP_PREFIX_ROUTE
        route_dist = '65000:100'
        ethernet_tag_id = 200
        ip_prefix = '192.168.0.0/24'
        expected_kwargs = {'route_type': route_type, 'route_dist': route_dist, 'ethernet_tag_id': ethernet_tag_id, 'ip_prefix': ip_prefix}
        speaker = bgpspeaker.BGPSpeaker(65000, '10.0.0.1')
        speaker.evpn_prefix_del(route_type=route_type, route_dist=route_dist, ethernet_tag_id=ethernet_tag_id, ip_prefix=ip_prefix)
        mock_call.assert_called_with('evpn_prefix.delete_local', **expected_kwargs)

    @mock.patch('os_ken.services.protocols.bgp.bgpspeaker.BGPSpeaker.__init__', mock.MagicMock(return_value=None))
    @mock.patch('os_ken.services.protocols.bgp.bgpspeaker.call')
    def test_evpn_prefix_add_pmsi_no_tunnel_info(self, mock_call):
        route_type = bgpspeaker.EVPN_MULTICAST_ETAG_ROUTE
        route_dist = '65000:100'
        ethernet_tag_id = 200
        next_hop = '0.0.0.0'
        ip_addr = '192.168.0.1'
        pmsi_tunnel_type = bgpspeaker.PMSI_TYPE_NO_TUNNEL_INFO
        expected_kwargs = {'route_type': route_type, 'route_dist': route_dist, 'ethernet_tag_id': ethernet_tag_id, 'next_hop': next_hop, 'ip_addr': ip_addr, 'pmsi_tunnel_type': pmsi_tunnel_type}
        speaker = bgpspeaker.BGPSpeaker(65000, '10.0.0.1')
        speaker.evpn_prefix_add(route_type=route_type, route_dist=route_dist, ethernet_tag_id=ethernet_tag_id, ip_addr=ip_addr, pmsi_tunnel_type=pmsi_tunnel_type)
        mock_call.assert_called_with('evpn_prefix.add_local', **expected_kwargs)

    @mock.patch('os_ken.services.protocols.bgp.bgpspeaker.BGPSpeaker.__init__', mock.MagicMock(return_value=None))
    @mock.patch('os_ken.services.protocols.bgp.bgpspeaker.call')
    def test_evpn_prefix_add_pmsi_ingress_rep(self, mock_call):
        route_type = bgpspeaker.EVPN_MULTICAST_ETAG_ROUTE
        route_dist = '65000:100'
        ethernet_tag_id = 200
        next_hop = '0.0.0.0'
        ip_addr = '192.168.0.1'
        pmsi_tunnel_type = bgpspeaker.PMSI_TYPE_INGRESS_REP
        expected_kwargs = {'route_type': route_type, 'route_dist': route_dist, 'ethernet_tag_id': ethernet_tag_id, 'next_hop': next_hop, 'ip_addr': ip_addr, 'pmsi_tunnel_type': pmsi_tunnel_type}
        speaker = bgpspeaker.BGPSpeaker(65000, '10.0.0.1')
        speaker.evpn_prefix_add(route_type=route_type, route_dist=route_dist, ethernet_tag_id=ethernet_tag_id, ip_addr=ip_addr, pmsi_tunnel_type=pmsi_tunnel_type)
        mock_call.assert_called_with('evpn_prefix.add_local', **expected_kwargs)

    @mock.patch('os_ken.services.protocols.bgp.bgpspeaker.BGPSpeaker.__init__', mock.MagicMock(return_value=None))
    @mock.patch('os_ken.services.protocols.bgp.bgpspeaker.call')
    def test_evpn_prefix_add_invalid_pmsi_tunnel_type(self, mock_call):
        route_type = bgpspeaker.EVPN_MULTICAST_ETAG_ROUTE
        route_dist = '65000:100'
        ethernet_tag_id = 200
        next_hop = '0.0.0.0'
        ip_addr = '192.168.0.1'
        pmsi_tunnel_type = 1
        speaker = bgpspeaker.BGPSpeaker(65000, '10.0.0.1')
        self.assertRaises(ValueError, speaker.evpn_prefix_add, route_type=route_type, route_dist=route_dist, ethernet_tag_id=ethernet_tag_id, ip_addr=ip_addr, pmsi_tunnel_type=pmsi_tunnel_type)

    @mock.patch('os_ken.services.protocols.bgp.bgpspeaker.BGPSpeaker.__init__', mock.MagicMock(return_value=None))
    @mock.patch('os_ken.services.protocols.bgp.bgpspeaker.call')
    def test_flowspec_prefix_add_ipv4(self, mock_call):
        flowspec_family = bgpspeaker.FLOWSPEC_FAMILY_IPV4
        rules = {'dst_prefix': '10.60.1.0/24'}
        actions = {'traffic_marking': {'dscp': 24}}
        expected_kwargs = {'flowspec_family': flowspec_family, 'rules': rules, 'actions': actions}
        speaker = bgpspeaker.BGPSpeaker(65000, '10.0.0.1')
        speaker.flowspec_prefix_add(flowspec_family=flowspec_family, rules=rules, actions=actions)
        mock_call.assert_called_with('flowspec.add', **expected_kwargs)

    @mock.patch('os_ken.services.protocols.bgp.bgpspeaker.BGPSpeaker.__init__', mock.MagicMock(return_value=None))
    @mock.patch('os_ken.services.protocols.bgp.bgpspeaker.call')
    def test_flowspec_prefix_add_ipv4_without_actions(self, mock_call):
        flowspec_family = bgpspeaker.FLOWSPEC_FAMILY_IPV4
        rules = {'dst_prefix': '10.60.1.0/24'}
        expected_kwargs = {'flowspec_family': flowspec_family, 'rules': rules, 'actions': {}}
        speaker = bgpspeaker.BGPSpeaker(65000, '10.0.0.1')
        speaker.flowspec_prefix_add(flowspec_family=flowspec_family, rules=rules)
        mock_call.assert_called_with('flowspec.add', **expected_kwargs)

    @mock.patch('os_ken.services.protocols.bgp.bgpspeaker.BGPSpeaker.__init__', mock.MagicMock(return_value=None))
    @mock.patch('os_ken.services.protocols.bgp.bgpspeaker.call')
    def test_flowspec_prefix_del_ipv4(self, mock_call):
        flowspec_family = bgpspeaker.FLOWSPEC_FAMILY_IPV4
        rules = {'dst_prefix': '10.60.1.0/24'}
        expected_kwargs = {'flowspec_family': flowspec_family, 'rules': rules}
        speaker = bgpspeaker.BGPSpeaker(65000, '10.0.0.1')
        speaker.flowspec_prefix_del(flowspec_family=flowspec_family, rules=rules)
        mock_call.assert_called_with('flowspec.del', **expected_kwargs)

    @mock.patch('os_ken.services.protocols.bgp.bgpspeaker.BGPSpeaker.__init__', mock.MagicMock(return_value=None))
    @mock.patch('os_ken.services.protocols.bgp.bgpspeaker.call')
    def test_flowspec_prefix_add_vpnv4(self, mock_call):
        flowspec_family = bgpspeaker.FLOWSPEC_FAMILY_VPNV4
        route_dist = '65001:100'
        rules = {'dst_prefix': '10.70.1.0/24'}
        actions = {'traffic_marking': {'dscp': 24}}
        expected_kwargs = {'flowspec_family': flowspec_family, 'route_dist': route_dist, 'rules': rules, 'actions': actions}
        speaker = bgpspeaker.BGPSpeaker(65000, '10.0.0.1')
        speaker.flowspec_prefix_add(flowspec_family=flowspec_family, route_dist=route_dist, rules=rules, actions=actions)
        mock_call.assert_called_with('flowspec.add_local', **expected_kwargs)

    @mock.patch('os_ken.services.protocols.bgp.bgpspeaker.BGPSpeaker.__init__', mock.MagicMock(return_value=None))
    @mock.patch('os_ken.services.protocols.bgp.bgpspeaker.call')
    def test_flowspec_prefix_del_vpnv4(self, mock_call):
        flowspec_family = bgpspeaker.FLOWSPEC_FAMILY_VPNV4
        route_dist = '65001:100'
        rules = {'dst_prefix': '10.70.1.0/24'}
        expected_kwargs = {'flowspec_family': flowspec_family, 'route_dist': route_dist, 'rules': rules}
        speaker = bgpspeaker.BGPSpeaker(65000, '10.0.0.1')
        speaker.flowspec_prefix_del(flowspec_family=flowspec_family, route_dist=route_dist, rules=rules)
        mock_call.assert_called_with('flowspec.del_local', **expected_kwargs)

    @mock.patch('os_ken.services.protocols.bgp.bgpspeaker.BGPSpeaker.__init__', mock.MagicMock(return_value=None))
    @mock.patch('os_ken.services.protocols.bgp.bgpspeaker.call')
    def test_flowspec_prefix_add_ipv6(self, mock_call):
        flowspec_family = bgpspeaker.FLOWSPEC_FAMILY_IPV6
        rules = {'dst_prefix': '2001::3/128/32'}
        actions = {'traffic_marking': {'dscp': 24}}
        expected_kwargs = {'flowspec_family': flowspec_family, 'rules': rules, 'actions': actions}
        speaker = bgpspeaker.BGPSpeaker(65000, '10.0.0.1')
        speaker.flowspec_prefix_add(flowspec_family=flowspec_family, rules=rules, actions=actions)
        mock_call.assert_called_with('flowspec.add', **expected_kwargs)

    @mock.patch('os_ken.services.protocols.bgp.bgpspeaker.BGPSpeaker.__init__', mock.MagicMock(return_value=None))
    @mock.patch('os_ken.services.protocols.bgp.bgpspeaker.call')
    def test_flowspec_prefix_add_ipv6_without_actions(self, mock_call):
        flowspec_family = bgpspeaker.FLOWSPEC_FAMILY_IPV6
        rules = {'dst_prefix': '2001::3/128/32'}
        expected_kwargs = {'flowspec_family': flowspec_family, 'rules': rules, 'actions': {}}
        speaker = bgpspeaker.BGPSpeaker(65000, '10.0.0.1')
        speaker.flowspec_prefix_add(flowspec_family=flowspec_family, rules=rules)
        mock_call.assert_called_with('flowspec.add', **expected_kwargs)

    @mock.patch('os_ken.services.protocols.bgp.bgpspeaker.BGPSpeaker.__init__', mock.MagicMock(return_value=None))
    @mock.patch('os_ken.services.protocols.bgp.bgpspeaker.call')
    def test_flowspec_prefix_del_ipv6(self, mock_call):
        flowspec_family = bgpspeaker.FLOWSPEC_FAMILY_IPV6
        rules = {'dst_prefix': '2001::3/128/32'}
        expected_kwargs = {'flowspec_family': flowspec_family, 'rules': rules}
        speaker = bgpspeaker.BGPSpeaker(65000, '10.0.0.1')
        speaker.flowspec_prefix_del(flowspec_family=flowspec_family, rules=rules)
        mock_call.assert_called_with('flowspec.del', **expected_kwargs)

    @mock.patch('os_ken.services.protocols.bgp.bgpspeaker.BGPSpeaker.__init__', mock.MagicMock(return_value=None))
    @mock.patch('os_ken.services.protocols.bgp.bgpspeaker.call')
    def test_flowspec_prefix_add_vpnv6(self, mock_call):
        flowspec_family = bgpspeaker.FLOWSPEC_FAMILY_VPNV6
        route_dist = '65001:100'
        rules = {'dst_prefix': '2001::3/128/32'}
        actions = {'traffic_marking': {'dscp': 24}}
        expected_kwargs = {'flowspec_family': flowspec_family, 'route_dist': route_dist, 'rules': rules, 'actions': actions}
        speaker = bgpspeaker.BGPSpeaker(65000, '10.0.0.1')
        speaker.flowspec_prefix_add(flowspec_family=flowspec_family, route_dist=route_dist, rules=rules, actions=actions)
        mock_call.assert_called_with('flowspec.add_local', **expected_kwargs)

    @mock.patch('os_ken.services.protocols.bgp.bgpspeaker.BGPSpeaker.__init__', mock.MagicMock(return_value=None))
    @mock.patch('os_ken.services.protocols.bgp.bgpspeaker.call')
    def test_flowspec_prefix_del_vpnv6(self, mock_call):
        flowspec_family = bgpspeaker.FLOWSPEC_FAMILY_VPNV6
        route_dist = '65001:100'
        rules = {'dst_prefix': '2001::3/128/32'}
        expected_kwargs = {'flowspec_family': flowspec_family, 'route_dist': route_dist, 'rules': rules}
        speaker = bgpspeaker.BGPSpeaker(65000, '10.0.0.1')
        speaker.flowspec_prefix_del(flowspec_family=flowspec_family, route_dist=route_dist, rules=rules)
        mock_call.assert_called_with('flowspec.del_local', **expected_kwargs)

    @mock.patch('os_ken.services.protocols.bgp.bgpspeaker.BGPSpeaker.__init__', mock.MagicMock(return_value=None))
    @mock.patch('os_ken.services.protocols.bgp.bgpspeaker.call')
    def test_flowspec_prefix_add_l2vpn(self, mock_call):
        flowspec_family = bgpspeaker.FLOWSPEC_FAMILY_L2VPN
        route_dist = '65001:100'
        rules = {'dst_mac': '12:34:56:78:9a:bc'}
        actions = {'traffic_marking': {'dscp': 24}}
        expected_kwargs = {'flowspec_family': flowspec_family, 'route_dist': route_dist, 'rules': rules, 'actions': actions}
        speaker = bgpspeaker.BGPSpeaker(65000, '10.0.0.1')
        speaker.flowspec_prefix_add(flowspec_family=flowspec_family, route_dist=route_dist, rules=rules, actions=actions)
        mock_call.assert_called_with('flowspec.add_local', **expected_kwargs)

    @mock.patch('os_ken.services.protocols.bgp.bgpspeaker.BGPSpeaker.__init__', mock.MagicMock(return_value=None))
    @mock.patch('os_ken.services.protocols.bgp.bgpspeaker.call')
    def test_flowspec_prefix_del_l2vpn(self, mock_call):
        flowspec_family = bgpspeaker.FLOWSPEC_FAMILY_L2VPN
        route_dist = '65001:100'
        rules = {'dst_mac': '12:34:56:78:9a:bc'}
        expected_kwargs = {'flowspec_family': flowspec_family, 'route_dist': route_dist, 'rules': rules}
        speaker = bgpspeaker.BGPSpeaker(65000, '10.0.0.1')
        speaker.flowspec_prefix_del(flowspec_family=flowspec_family, route_dist=route_dist, rules=rules)
        mock_call.assert_called_with('flowspec.del_local', **expected_kwargs)