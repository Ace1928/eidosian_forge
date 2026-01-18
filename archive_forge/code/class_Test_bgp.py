import logging
import os
import sys
import unittest
from os_ken.utils import binary_str
from os_ken.lib import pcaplib
from os_ken.lib.packet import packet
from os_ken.lib.packet import bgp
from os_ken.lib.packet import afi
from os_ken.lib.packet import safi
class Test_bgp(unittest.TestCase):
    """ Test case for os_ken.lib.packet.bgp
    """

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_open1(self):
        msg = bgp.BGPOpen(my_as=30000, bgp_identifier='192.0.2.1')
        binmsg = msg.serialize()
        msg2, _, rest = bgp.BGPMessage.parser(binmsg)
        self.assertEqual(str(msg), str(msg2))
        self.assertEqual(len(msg), 29)
        self.assertEqual(rest, b'')

    def test_open2(self):
        opt_param = [bgp.BGPOptParamCapabilityUnknown(cap_code=200, cap_value=b'hoge'), bgp.BGPOptParamCapabilityGracefulRestart(flags=0, time=120, tuples=[]), bgp.BGPOptParamCapabilityRouteRefresh(), bgp.BGPOptParamCapabilityCiscoRouteRefresh(), bgp.BGPOptParamCapabilityMultiprotocol(afi=afi.IP, safi=safi.MPLS_VPN), bgp.BGPOptParamCapabilityCarryingLabelInfo(), bgp.BGPOptParamCapabilityFourOctetAsNumber(as_number=1234567), bgp.BGPOptParamUnknown(type_=99, value=b'fuga')]
        msg = bgp.BGPOpen(my_as=30000, bgp_identifier='192.0.2.2', opt_param=opt_param)
        binmsg = msg.serialize()
        msg2, _, rest = bgp.BGPMessage.parser(binmsg)
        self.assertEqual(str(msg), str(msg2))
        self.assertTrue(len(msg) > 29)
        self.assertEqual(rest, b'')

    def test_update1(self):
        msg = bgp.BGPUpdate()
        binmsg = msg.serialize()
        msg2, _, rest = bgp.BGPMessage.parser(binmsg)
        self.assertEqual(str(msg), str(msg2))
        self.assertEqual(len(msg), 23)
        self.assertEqual(rest, b'')

    def test_update2(self):
        withdrawn_routes = [bgp.BGPWithdrawnRoute(length=0, addr='192.0.2.13'), bgp.BGPWithdrawnRoute(length=1, addr='192.0.2.13'), bgp.BGPWithdrawnRoute(length=3, addr='192.0.2.13'), bgp.BGPWithdrawnRoute(length=7, addr='192.0.2.13'), bgp.BGPWithdrawnRoute(length=32, addr='192.0.2.13')]
        mp_nlri = [bgp.LabelledVPNIPAddrPrefix(24, '192.0.9.0', route_dist='100:100', labels=[1, 2, 3]), bgp.LabelledVPNIPAddrPrefix(26, '192.0.10.192', route_dist='10.0.0.1:10000', labels=[5, 6, 7, 8])]
        mp_nlri2 = [bgp.LabelledIPAddrPrefix(24, '192.168.0.0', labels=[1, 2, 3])]
        mp_nlri_v6 = [bgp.LabelledVPNIP6AddrPrefix(64, '2001:db8:1111::', route_dist='200:200', labels=[1, 2, 3]), bgp.LabelledVPNIP6AddrPrefix(64, '2001:db8:2222::', route_dist='10.0.0.1:10000', labels=[5, 6, 7, 8])]
        mp_nlri2_v6 = [bgp.LabelledIP6AddrPrefix(64, '2001:db8:3333::', labels=[1, 2, 3])]
        communities = [bgp.BGP_COMMUNITY_NO_EXPORT, bgp.BGP_COMMUNITY_NO_ADVERTISE]
        ecommunities = [bgp.BGPTwoOctetAsSpecificExtendedCommunity(subtype=1, as_number=65500, local_administrator=3908876543), bgp.BGPFourOctetAsSpecificExtendedCommunity(subtype=2, as_number=10000000, local_administrator=59876), bgp.BGPIPv4AddressSpecificExtendedCommunity(subtype=3, ipv4_address='192.0.2.1', local_administrator=65432), bgp.BGPOpaqueExtendedCommunity(subtype=13, opaque=b'abcdef'), bgp.BGPEncapsulationExtendedCommunity(subtype=12, tunnel_type=10), bgp.BGPEvpnMacMobilityExtendedCommunity(subtype=0, flags=255, sequence_number=287454020), bgp.BGPEvpnEsiLabelExtendedCommunity(subtype=1, flags=255, label=b'\xff\xff\xff'), bgp.BGPEvpnEsiLabelExtendedCommunity(subtype=1, flags=255, mpls_label=1048575), bgp.BGPEvpnEsiLabelExtendedCommunity(subtype=1, flags=255, vni=16777215), bgp.BGPEvpnEsImportRTExtendedCommunity(subtype=2, es_import='aa:bb:cc:dd:ee:ff'), bgp.BGPUnknownExtendedCommunity(type_=99, value=b'abcdefg')]
        path_attributes = [bgp.BGPPathAttributeOrigin(value=1), bgp.BGPPathAttributeAsPath(value=[[1000], {1001, 1002}, [1003, 1004]]), bgp.BGPPathAttributeNextHop(value='192.0.2.199'), bgp.BGPPathAttributeMultiExitDisc(value=2000000000), bgp.BGPPathAttributeLocalPref(value=1000000000), bgp.BGPPathAttributeAtomicAggregate(), bgp.BGPPathAttributeAggregator(as_number=40000, addr='192.0.2.99'), bgp.BGPPathAttributeCommunities(communities=communities), bgp.BGPPathAttributeOriginatorId(value='10.1.1.1'), bgp.BGPPathAttributeClusterList(value=['1.1.1.1', '2.2.2.2']), bgp.BGPPathAttributeExtendedCommunities(communities=ecommunities), bgp.BGPPathAttributePmsiTunnel(pmsi_flags=1, tunnel_type=PMSI_TYPE_NO_TUNNEL_INFORMATION_PRESENT, label=b'\xff\xff\xff'), bgp.BGPPathAttributePmsiTunnel(pmsi_flags=1, tunnel_type=PMSI_TYPE_NO_TUNNEL_INFORMATION_PRESENT, tunnel_id=None), bgp.BGPPathAttributePmsiTunnel(pmsi_flags=1, tunnel_type=PMSI_TYPE_INGRESS_REPLICATION, mpls_label=1048575, tunnel_id=bgp.PmsiTunnelIdIngressReplication(tunnel_endpoint_ip='1.1.1.1')), bgp.BGPPathAttributePmsiTunnel(pmsi_flags=1, tunnel_type=PMSI_TYPE_INGRESS_REPLICATION, vni=16777215, tunnel_id=bgp.PmsiTunnelIdIngressReplication(tunnel_endpoint_ip='aa:bb:cc::dd:ee:ff')), bgp.BGPPathAttributePmsiTunnel(pmsi_flags=1, tunnel_type=2, label=b'\xff\xff\xff', tunnel_id=bgp.PmsiTunnelIdUnknown(value=b'test')), bgp.BGPPathAttributeAs4Path(value=[[1000000], {1000001, 1002}, [1003, 1000004]]), bgp.BGPPathAttributeAs4Aggregator(as_number=100040000, addr='192.0.2.99'), bgp.BGPPathAttributeMpReachNLRI(afi=afi.IP, safi=safi.MPLS_VPN, next_hop='1.1.1.1', nlri=mp_nlri), bgp.BGPPathAttributeMpReachNLRI(afi=afi.IP, safi=safi.MPLS_LABEL, next_hop='1.1.1.1', nlri=mp_nlri2), bgp.BGPPathAttributeMpReachNLRI(afi=afi.IP6, safi=safi.MPLS_VPN, next_hop=['2001:db8::1'], nlri=mp_nlri_v6), bgp.BGPPathAttributeMpReachNLRI(afi=afi.IP6, safi=safi.MPLS_LABEL, next_hop=['2001:db8::1', 'fe80::1'], nlri=mp_nlri2_v6), bgp.BGPPathAttributeMpUnreachNLRI(afi=afi.IP, safi=safi.MPLS_VPN, withdrawn_routes=mp_nlri), bgp.BGPPathAttributeUnknown(flags=0, type_=100, value=300 * b'bar')]
        nlri = [bgp.BGPNLRI(length=24, addr='203.0.113.1'), bgp.BGPNLRI(length=16, addr='203.0.113.0')]
        msg = bgp.BGPUpdate(withdrawn_routes=withdrawn_routes, path_attributes=path_attributes, nlri=nlri)
        binmsg = msg.serialize()
        msg2, _, rest = bgp.BGPMessage.parser(binmsg)
        self.assertEqual(str(msg), str(msg2))
        self.assertTrue(len(msg) > 23)
        self.assertEqual(rest, b'')

    def test_keepalive(self):
        msg = bgp.BGPKeepAlive()
        binmsg = msg.serialize()
        msg2, _, rest = bgp.BGPMessage.parser(binmsg)
        self.assertEqual(str(msg), str(msg2))
        self.assertEqual(len(msg), 19)
        self.assertEqual(rest, b'')

    def test_notification(self):
        data = b'hoge'
        msg = bgp.BGPNotification(error_code=1, error_subcode=2, data=data)
        binmsg = msg.serialize()
        msg2, _, rest = bgp.BGPMessage.parser(binmsg)
        self.assertEqual(str(msg), str(msg2))
        self.assertEqual(len(msg), 21 + len(data))
        self.assertEqual(rest, b'')

    def test_route_refresh(self):
        msg = bgp.BGPRouteRefresh(afi=afi.IP, safi=safi.MPLS_VPN)
        binmsg = msg.serialize()
        msg2, _, rest = bgp.BGPMessage.parser(binmsg)
        self.assertEqual(str(msg), str(msg2))
        self.assertEqual(len(msg), 23)
        self.assertEqual(rest, b'')

    def test_stream_parser(self):
        msgs = [bgp.BGPNotification(error_code=1, error_subcode=2, data=b'foo'), bgp.BGPNotification(error_code=3, error_subcode=4, data=b'bar'), bgp.BGPNotification(error_code=5, error_subcode=6, data=b'baz')]
        binmsgs = b''.join([bytes(msg.serialize()) for msg in msgs])
        sp = bgp.StreamParser()
        results = []
        for b in binmsgs:
            for m in sp.parse(b):
                results.append(m)
        self.assertEqual(str(results), str(msgs))

    def test_parser(self):
        files = ['bgp4-open', 'bgp4-update', 'bgp4-update_ipv6', 'bgp4-update_vpnv6', 'bgp4-keepalive', 'evpn_esi_arbitrary', 'evpn_esi_lacp', 'evpn_esi_l2_bridge', 'evpn_esi_mac_base', 'evpn_esi_router_id', 'evpn_esi_as_based', 'evpn_nlri_eth_a-d', 'evpn_nlri_mac_ip_ad', 'evpn_nlri_inc_multi_eth_tag', 'evpn_nlri_eth_seg', 'evpn_nlri_ip_prefix', 'flowspec_nlri_ipv4', 'flowspec_nlri_vpn4', 'flowspec_nlri_ipv6', 'flowspec_nlri_vpn6', 'flowspec_nlri_l2vpn', 'flowspec_action_traffic_rate', 'flowspec_action_traffic_action', 'flowspec_action_redirect', 'flowspec_action_traffic_marking']
        for f in files:
            LOG.debug('*** testing %s ...', f)
            for _, buf in pcaplib.Reader(open(BGP4_PACKET_DATA_DIR + f + '.pcap', 'rb')):
                pkt = packet.Packet(buf)
                self.assertTrue(isinstance(pkt.protocols[-1], bgp.BGPMessage), 'Failed to parse BGP message: %s' % pkt)
                pkt.serialize()
                self.assertEqual(buf, pkt.data, "b'%s' != b'%s'" % (binary_str(buf), binary_str(pkt.data)))

    def test_vlan_action_parser(self):
        action = bgp.BGPFlowSpecVlanActionCommunity(actions_1=bgp.BGPFlowSpecVlanActionCommunity.POP | bgp.BGPFlowSpecVlanActionCommunity.SWAP, vlan_1=3000, cos_1=3, actions_2=bgp.BGPFlowSpecVlanActionCommunity.PUSH, vlan_2=4000, cos_2=2)
        binmsg = action.serialize()
        msg, rest = bgp.BGPFlowSpecVlanActionCommunity.parse(binmsg)
        self.assertEqual(str(action), str(msg))
        self.assertEqual(rest, b'')

    def test_tpid_action_parser(self):
        action = bgp.BGPFlowSpecTPIDActionCommunity(actions=bgp.BGPFlowSpecTPIDActionCommunity.TI | bgp.BGPFlowSpecTPIDActionCommunity.TO, tpid_1=5, tpid_2=6)
        binmsg = action.serialize()
        msg, rest = bgp.BGPFlowSpecTPIDActionCommunity.parse(binmsg)
        self.assertEqual(str(action), str(msg))
        self.assertEqual(rest, b'')

    def test_json1(self):
        opt_param = [bgp.BGPOptParamCapabilityUnknown(cap_code=200, cap_value=b'hoge'), bgp.BGPOptParamCapabilityRouteRefresh(), bgp.BGPOptParamCapabilityMultiprotocol(afi=afi.IP, safi=safi.MPLS_VPN), bgp.BGPOptParamCapabilityFourOctetAsNumber(as_number=1234567), bgp.BGPOptParamUnknown(type_=99, value=b'fuga')]
        msg1 = bgp.BGPOpen(my_as=30000, bgp_identifier='192.0.2.2', opt_param=opt_param)
        jsondict = msg1.to_jsondict()
        msg2 = bgp.BGPOpen.from_jsondict(jsondict['BGPOpen'])
        self.assertEqual(str(msg1), str(msg2))

    def test_json2(self):
        withdrawn_routes = [bgp.BGPWithdrawnRoute(length=0, addr='192.0.2.13'), bgp.BGPWithdrawnRoute(length=1, addr='192.0.2.13'), bgp.BGPWithdrawnRoute(length=3, addr='192.0.2.13'), bgp.BGPWithdrawnRoute(length=7, addr='192.0.2.13'), bgp.BGPWithdrawnRoute(length=32, addr='192.0.2.13')]
        mp_nlri = [bgp.LabelledVPNIPAddrPrefix(24, '192.0.9.0', route_dist='100:100', labels=[1, 2, 3]), bgp.LabelledVPNIPAddrPrefix(26, '192.0.10.192', route_dist='10.0.0.1:10000', labels=[5, 6, 7, 8])]
        mp_nlri2 = [bgp.LabelledIPAddrPrefix(24, '192.168.0.0', labels=[1, 2, 3])]
        mp_nlri_v6 = [bgp.LabelledVPNIP6AddrPrefix(64, '2001:db8:1111::', route_dist='200:200', labels=[1, 2, 3]), bgp.LabelledVPNIP6AddrPrefix(64, '2001:db8:2222::', route_dist='10.0.0.1:10000', labels=[5, 6, 7, 8])]
        mp_nlri2_v6 = [bgp.LabelledIP6AddrPrefix(64, '2001:db8:3333::', labels=[1, 2, 3])]
        communities = [bgp.BGP_COMMUNITY_NO_EXPORT, bgp.BGP_COMMUNITY_NO_ADVERTISE]
        ecommunities = [bgp.BGPTwoOctetAsSpecificExtendedCommunity(subtype=1, as_number=65500, local_administrator=3908876543), bgp.BGPFourOctetAsSpecificExtendedCommunity(subtype=2, as_number=10000000, local_administrator=59876), bgp.BGPIPv4AddressSpecificExtendedCommunity(subtype=3, ipv4_address='192.0.2.1', local_administrator=65432), bgp.BGPOpaqueExtendedCommunity(subtype=13, opaque=b'abcdef'), bgp.BGPEncapsulationExtendedCommunity(subtype=12, tunnel_type=10), bgp.BGPEvpnMacMobilityExtendedCommunity(subtype=0, flags=255, sequence_number=287454020), bgp.BGPEvpnEsiLabelExtendedCommunity(subtype=1, flags=255, label=b'\xff\xff\xff'), bgp.BGPEvpnEsiLabelExtendedCommunity(subtype=1, flags=255, mpls_label=1048575), bgp.BGPEvpnEsiLabelExtendedCommunity(subtype=1, flags=255, vni=16777215), bgp.BGPEvpnEsImportRTExtendedCommunity(subtype=2, es_import='aa:bb:cc:dd:ee:ff'), bgp.BGPUnknownExtendedCommunity(type_=99, value=b'abcdefg')]
        path_attributes = [bgp.BGPPathAttributeOrigin(value=1), bgp.BGPPathAttributeAsPath(value=[[1000], {1001, 1002}, [1003, 1004]]), bgp.BGPPathAttributeNextHop(value='192.0.2.199'), bgp.BGPPathAttributeMultiExitDisc(value=2000000000), bgp.BGPPathAttributeLocalPref(value=1000000000), bgp.BGPPathAttributeAtomicAggregate(), bgp.BGPPathAttributeAggregator(as_number=40000, addr='192.0.2.99'), bgp.BGPPathAttributeCommunities(communities=communities), bgp.BGPPathAttributeExtendedCommunities(communities=ecommunities), bgp.BGPPathAttributePmsiTunnel(pmsi_flags=1, tunnel_type=PMSI_TYPE_NO_TUNNEL_INFORMATION_PRESENT, label=b'\xff\xff\xff'), bgp.BGPPathAttributePmsiTunnel(pmsi_flags=1, tunnel_type=PMSI_TYPE_NO_TUNNEL_INFORMATION_PRESENT, tunnel_id=None), bgp.BGPPathAttributePmsiTunnel(pmsi_flags=1, tunnel_type=PMSI_TYPE_INGRESS_REPLICATION, mpls_label=1048575, tunnel_id=bgp.PmsiTunnelIdIngressReplication(tunnel_endpoint_ip='1.1.1.1')), bgp.BGPPathAttributePmsiTunnel(pmsi_flags=1, tunnel_type=PMSI_TYPE_INGRESS_REPLICATION, vni=16777215, tunnel_id=bgp.PmsiTunnelIdIngressReplication(tunnel_endpoint_ip='aa:bb:cc::dd:ee:ff')), bgp.BGPPathAttributePmsiTunnel(pmsi_flags=1, tunnel_type=2, label=b'\xff\xff\xff', tunnel_id=bgp.PmsiTunnelIdUnknown(value=b'test')), bgp.BGPPathAttributeAs4Path(value=[[1000000], {1000001, 1002}, [1003, 1000004]]), bgp.BGPPathAttributeAs4Aggregator(as_number=100040000, addr='192.0.2.99'), bgp.BGPPathAttributeMpReachNLRI(afi=afi.IP, safi=safi.MPLS_VPN, next_hop='1.1.1.1', nlri=mp_nlri), bgp.BGPPathAttributeMpReachNLRI(afi=afi.IP, safi=safi.MPLS_LABEL, next_hop='1.1.1.1', nlri=mp_nlri2), bgp.BGPPathAttributeMpReachNLRI(afi=afi.IP6, safi=safi.MPLS_VPN, next_hop=['2001:db8::1'], nlri=mp_nlri_v6), bgp.BGPPathAttributeMpReachNLRI(afi=afi.IP6, safi=safi.MPLS_LABEL, next_hop=['2001:db8::1', 'fe80::1'], nlri=mp_nlri2_v6), bgp.BGPPathAttributeMpUnreachNLRI(afi=afi.IP, safi=safi.MPLS_VPN, withdrawn_routes=mp_nlri), bgp.BGPPathAttributeUnknown(flags=0, type_=100, value=300 * b'bar')]
        nlri = [bgp.BGPNLRI(length=24, addr='203.0.113.1'), bgp.BGPNLRI(length=16, addr='203.0.113.0')]
        msg1 = bgp.BGPUpdate(withdrawn_routes=withdrawn_routes, path_attributes=path_attributes, nlri=nlri)
        jsondict = msg1.to_jsondict()
        msg2 = bgp.BGPUpdate.from_jsondict(jsondict['BGPUpdate'])
        self.assertEqual(str(msg1), str(msg2))

    def test_flowspec_user_interface_ipv4(self):
        rules = RULES_BASE + [bgp.FlowSpecDestPrefix(addr='10.0.0.0', length=24), bgp.FlowSpecSrcPrefix(addr='20.0.0.0', length=24), bgp.FlowSpecIPProtocol(operator=bgp.FlowSpecIPProtocol.EQ, value=6), bgp.FlowSpecFragment(operator=0, value=bgp.FlowSpecFragment.LF), bgp.FlowSpecFragment(operator=bgp.FlowSpecFragment.MATCH, value=bgp.FlowSpecFragment.FF), bgp.FlowSpecFragment(operator=bgp.FlowSpecFragment.AND | bgp.FlowSpecFragment.MATCH, value=bgp.FlowSpecFragment.ISF), bgp.FlowSpecFragment(operator=bgp.FlowSpecFragment.NOT, value=bgp.FlowSpecFragment.DF)]
        msg = bgp.FlowSpecIPv4NLRI.from_user(dst_prefix='10.0.0.0/24', src_prefix='20.0.0.0/24', ip_proto='6', port='>=8000 & <=9000 | ==80', dst_port='8080 >9000&<9050 | <=1000', src_port='<=9090 & >=9080 <10100 & >10000', icmp_type=0, icmp_code=6, tcp_flags='SYN+ACK & !=URGENT', packet_len='1000 & 1100', dscp='22 24', fragment='LF ==FF&==ISF | !=DF')
        msg2 = bgp.FlowSpecIPv4NLRI(rules=rules)
        binmsg = msg.serialize()
        binmsg2 = msg2.serialize()
        self.assertEqual(str(msg), str(msg2))
        self.assertEqual(binary_str(binmsg), binary_str(binmsg2))
        msg3, rest = bgp.FlowSpecIPv4NLRI.parser(binmsg)
        self.assertEqual(str(msg), str(msg3))
        self.assertEqual(rest, b'')

    def test_flowspec_user_interface_vpv4(self):
        rules = RULES_BASE + [bgp.FlowSpecDestPrefix(addr='10.0.0.0', length=24), bgp.FlowSpecSrcPrefix(addr='20.0.0.0', length=24), bgp.FlowSpecIPProtocol(operator=bgp.FlowSpecIPProtocol.EQ, value=6), bgp.FlowSpecFragment(operator=0, value=bgp.FlowSpecFragment.LF), bgp.FlowSpecFragment(operator=bgp.FlowSpecFragment.MATCH, value=bgp.FlowSpecFragment.FF), bgp.FlowSpecFragment(operator=bgp.FlowSpecFragment.AND | bgp.FlowSpecFragment.MATCH, value=bgp.FlowSpecFragment.ISF), bgp.FlowSpecFragment(operator=bgp.FlowSpecFragment.NOT, value=bgp.FlowSpecFragment.DF)]
        msg = bgp.FlowSpecVPNv4NLRI.from_user(route_dist='65001:250', dst_prefix='10.0.0.0/24', src_prefix='20.0.0.0/24', ip_proto='6', port='>=8000 & <=9000 | ==80', dst_port='8080 >9000&<9050 | <=1000', src_port='<=9090 & >=9080 <10100 & >10000', icmp_type=0, icmp_code=6, tcp_flags='SYN+ACK & !=URGENT', packet_len='1000 & 1100', dscp='22 24', fragment='LF ==FF&==ISF | !=DF')
        msg2 = bgp.FlowSpecVPNv4NLRI(route_dist='65001:250', rules=rules)
        binmsg = msg.serialize()
        binmsg2 = msg2.serialize()
        self.assertEqual(str(msg), str(msg2))
        self.assertEqual(binary_str(binmsg), binary_str(binmsg2))
        msg3, rest = bgp.FlowSpecVPNv4NLRI.parser(binmsg)
        self.assertEqual(str(msg), str(msg3))
        self.assertEqual(rest, b'')

    def test_flowspec_user_interface_ipv6(self):
        rules = RULES_BASE + [bgp.FlowSpecIPv6DestPrefix(addr='2001::2', offset=32, length=128), bgp.FlowSpecIPv6SrcPrefix(addr='3002::3', length=128), bgp.FlowSpecNextHeader(operator=bgp.FlowSpecNextHeader.EQ, value=6), bgp.FlowSpecIPv6Fragment(operator=0, value=bgp.FlowSpecFragment.LF), bgp.FlowSpecIPv6Fragment(operator=bgp.FlowSpecFragment.MATCH, value=bgp.FlowSpecFragment.FF), bgp.FlowSpecIPv6Fragment(operator=bgp.FlowSpecFragment.AND | bgp.FlowSpecFragment.MATCH, value=bgp.FlowSpecFragment.ISF), bgp.FlowSpecIPv6Fragment(operator=bgp.FlowSpecFragment.NOT, value=bgp.FlowSpecFragment.LF), bgp.FlowSpecIPv6FlowLabel(operator=bgp.FlowSpecIPv6FlowLabel.EQ, value=100)]
        msg = bgp.FlowSpecIPv6NLRI.from_user(dst_prefix='2001::2/128/32', src_prefix='3002::3/128', next_header='6', port='>=8000 & <=9000 | ==80', dst_port='8080 >9000&<9050 | <=1000', src_port='<=9090 & >=9080 <10100 & >10000', icmp_type=0, icmp_code=6, tcp_flags='SYN+ACK & !=URGENT', packet_len='1000 & 1100', dscp='22 24', fragment='LF ==FF&==ISF | !=LF', flow_label=100)
        msg2 = bgp.FlowSpecIPv6NLRI(rules=rules)
        binmsg = msg.serialize()
        binmsg2 = msg2.serialize()
        self.assertEqual(str(msg), str(msg2))
        self.assertEqual(binary_str(binmsg), binary_str(binmsg2))
        msg3, rest = bgp.FlowSpecIPv6NLRI.parser(binmsg)
        self.assertEqual(str(msg), str(msg3))
        self.assertEqual(rest, b'')

    def test_flowspec_user_interface_vpnv6(self):
        rules = RULES_BASE + [bgp.FlowSpecIPv6DestPrefix(addr='2001::2', offset=32, length=128), bgp.FlowSpecIPv6SrcPrefix(addr='3002::3', length=128), bgp.FlowSpecNextHeader(operator=bgp.FlowSpecNextHeader.EQ, value=6), bgp.FlowSpecIPv6Fragment(operator=0, value=bgp.FlowSpecFragment.LF), bgp.FlowSpecIPv6Fragment(operator=bgp.FlowSpecFragment.MATCH, value=bgp.FlowSpecFragment.FF), bgp.FlowSpecIPv6Fragment(operator=bgp.FlowSpecFragment.AND | bgp.FlowSpecFragment.MATCH, value=bgp.FlowSpecFragment.ISF), bgp.FlowSpecIPv6Fragment(operator=bgp.FlowSpecFragment.NOT, value=bgp.FlowSpecFragment.LF), bgp.FlowSpecIPv6FlowLabel(operator=bgp.FlowSpecIPv6FlowLabel.EQ, value=100)]
        msg = bgp.FlowSpecVPNv6NLRI.from_user(route_dist='65001:250', dst_prefix='2001::2/128/32', src_prefix='3002::3/128', next_header='6', port='>=8000 & <=9000 | ==80', dst_port='8080 >9000&<9050 | <=1000', src_port='<=9090 & >=9080 <10100 & >10000', icmp_type=0, icmp_code=6, tcp_flags='SYN+ACK & !=URGENT', packet_len='1000 & 1100', dscp='22 24', fragment='LF ==FF&==ISF | !=LF', flow_label=100)
        msg2 = bgp.FlowSpecVPNv6NLRI(route_dist='65001:250', rules=rules)
        binmsg = msg.serialize()
        binmsg2 = msg2.serialize()
        self.assertEqual(str(msg), str(msg2))
        self.assertEqual(binary_str(binmsg), binary_str(binmsg2))
        msg3, rest = bgp.FlowSpecVPNv6NLRI.parser(binmsg)
        self.assertEqual(str(msg), str(msg3))
        self.assertEqual(rest, b'')

    def test_flowspec_user_interface_l2vpn(self):
        rules = RULES_L2VPN_BASE
        msg = bgp.FlowSpecL2VPNNLRI.from_user(route_dist='65001:250', ether_type=2048, src_mac='12:34:56:78:90:AB', dst_mac='BE:EF:C0:FF:EE:DD', llc_dsap=66, llc_ssap=66, llc_control=100, snap=74565, vlan_id='>4000', vlan_cos='>=3', inner_vlan_id='<3000', inner_vlan_cos='<=5')
        msg2 = bgp.FlowSpecL2VPNNLRI(route_dist='65001:250', rules=rules)
        binmsg = msg.serialize()
        binmsg2 = msg2.serialize()
        self.assertEqual(str(msg), str(msg2))
        self.assertEqual(binary_str(binmsg), binary_str(binmsg2))
        msg3, rest = bgp.FlowSpecL2VPNNLRI.parser(binmsg)
        self.assertEqual(str(msg), str(msg3))
        self.assertEqual(rest, b'')