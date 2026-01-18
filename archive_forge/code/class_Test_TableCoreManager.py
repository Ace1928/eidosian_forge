from collections import OrderedDict
import logging
import unittest
from unittest import mock
from os_ken.lib.packet.bgp import BGPPathAttributeOrigin
from os_ken.lib.packet.bgp import BGPPathAttributeAsPath
from os_ken.lib.packet.bgp import BGP_ATTR_ORIGIN_IGP
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_ORIGIN
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_AS_PATH
from os_ken.lib.packet.bgp import IPAddrPrefix
from os_ken.lib.packet.bgp import IP6AddrPrefix
from os_ken.lib.packet.bgp import EvpnArbitraryEsi
from os_ken.lib.packet.bgp import EvpnLACPEsi
from os_ken.lib.packet.bgp import EvpnEthernetAutoDiscoveryNLRI
from os_ken.lib.packet.bgp import EvpnMacIPAdvertisementNLRI
from os_ken.lib.packet.bgp import EvpnInclusiveMulticastEthernetTagNLRI
from os_ken.services.protocols.bgp.bgpspeaker import EVPN_MAX_ET
from os_ken.services.protocols.bgp.bgpspeaker import ESI_TYPE_LACP
from os_ken.services.protocols.bgp.core import BgpCoreError
from os_ken.services.protocols.bgp.core_managers import table_manager
from os_ken.services.protocols.bgp.rtconf.vrfs import VRF_RF_IPV4
from os_ken.services.protocols.bgp.rtconf.vrfs import VRF_RF_IPV6
from os_ken.services.protocols.bgp.rtconf.vrfs import VRF_RF_L2_EVPN
class Test_TableCoreManager(unittest.TestCase):
    """
    Test case for bgp.core_managers.table_manager.TableCoreManager
    """

    @mock.patch('os_ken.services.protocols.bgp.core_managers.TableCoreManager.__init__', mock.MagicMock(return_value=None))
    def _test_update_vrf_table(self, prefix_inst, route_dist, prefix_str, next_hop, route_family, route_type, is_withdraw=False, **kwargs):
        tbl_mng = table_manager.TableCoreManager(None, None)
        vrf_table_mock = mock.MagicMock()
        tbl_mng._tables = {(route_dist, route_family): vrf_table_mock}
        tbl_mng.update_vrf_table(route_dist=route_dist, prefix=prefix_str, next_hop=next_hop, route_family=route_family, route_type=route_type, is_withdraw=is_withdraw, **kwargs)
        call_args_list = vrf_table_mock.insert_vrf_path.call_args_list
        self.assertTrue(len(call_args_list) == 1)
        args, kwargs = call_args_list[0]
        self.assertTrue(len(args) == 0)
        self.assertEqual(str(prefix_inst), str(kwargs['nlri']))
        self.assertEqual(is_withdraw, kwargs['is_withdraw'])
        if is_withdraw:
            self.assertEqual(None, kwargs['next_hop'])
            self.assertEqual(False, kwargs['gen_lbl'])
        else:
            self.assertEqual(next_hop, kwargs['next_hop'])
            self.assertEqual(True, kwargs['gen_lbl'])

    def test_update_vrf_table_ipv4(self):
        route_dist = '65000:100'
        ip_network = '192.168.0.0'
        ip_prefix_len = 24
        prefix_str = '%s/%d' % (ip_network, ip_prefix_len)
        prefix_inst = IPAddrPrefix(ip_prefix_len, ip_network)
        next_hop = '10.0.0.1'
        route_family = VRF_RF_IPV4
        route_type = None
        kwargs = {}
        self._test_update_vrf_table(prefix_inst, route_dist, prefix_str, next_hop, route_family, route_type, **kwargs)

    def test_update_vrf_table_ipv6(self):
        route_dist = '65000:100'
        ip_network = 'fe80::'
        ip_prefix_len = 64
        prefix_str = '%s/%d' % (ip_network, ip_prefix_len)
        prefix_inst = IP6AddrPrefix(ip_prefix_len, ip_network)
        next_hop = 'fe80::0011:aabb:ccdd:eeff'
        route_family = VRF_RF_IPV6
        route_type = None
        kwargs = {}
        self._test_update_vrf_table(prefix_inst, route_dist, prefix_str, next_hop, route_family, route_type, **kwargs)

    def test_update_vrf_table_l2_evpn_with_esi_int(self):
        route_dist = '65000:100'
        prefix_str = None
        kwargs = {'ethernet_tag_id': 100, 'mac_addr': 'aa:bb:cc:dd:ee:ff', 'ip_addr': '192.168.0.1', 'mpls_labels': []}
        esi = EvpnArbitraryEsi(b'\x00\x00\x00\x00\x00\x00\x00\x00\x00')
        prefix_inst = EvpnMacIPAdvertisementNLRI(route_dist=route_dist, esi=esi, **kwargs)
        next_hop = '10.0.0.1'
        route_family = VRF_RF_L2_EVPN
        route_type = EvpnMacIPAdvertisementNLRI.ROUTE_TYPE_NAME
        kwargs['esi'] = 0
        self._test_update_vrf_table(prefix_inst, route_dist, prefix_str, next_hop, route_family, route_type, **kwargs)

    def test_update_vrf_table_l2_evpn_with_esi_dict(self):
        route_dist = '65000:100'
        prefix_str = None
        kwargs = {'ethernet_tag_id': EVPN_MAX_ET}
        esi = EvpnLACPEsi(mac_addr='aa:bb:cc:dd:ee:ff', port_key=100)
        prefix_inst = EvpnEthernetAutoDiscoveryNLRI(route_dist=route_dist, esi=esi, **kwargs)
        next_hop = '0.0.0.0'
        route_family = VRF_RF_L2_EVPN
        route_type = EvpnEthernetAutoDiscoveryNLRI.ROUTE_TYPE_NAME
        kwargs['esi'] = {'type': ESI_TYPE_LACP, 'mac_addr': 'aa:bb:cc:dd:ee:ff', 'port_key': 100}
        self._test_update_vrf_table(prefix_inst, route_dist, prefix_str, next_hop, route_family, route_type, **kwargs)

    def test_update_vrf_table_l2_evpn_without_esi(self):
        route_dist = '65000:100'
        prefix_str = None
        kwargs = {'ethernet_tag_id': 100, 'ip_addr': '192.168.0.1'}
        prefix_inst = EvpnInclusiveMulticastEthernetTagNLRI(route_dist=route_dist, **kwargs)
        next_hop = '10.0.0.1'
        route_family = VRF_RF_L2_EVPN
        route_type = EvpnInclusiveMulticastEthernetTagNLRI.ROUTE_TYPE_NAME
        self._test_update_vrf_table(prefix_inst, route_dist, prefix_str, next_hop, route_family, route_type, **kwargs)

    @mock.patch('os_ken.services.protocols.bgp.core_managers.TableCoreManager.__init__', mock.MagicMock(return_value=None))
    def test_update_vrf_table_l2_evpn_with_vni(self):
        route_dist = '65000:100'
        prefix_str = None
        kwargs = {'ethernet_tag_id': 100, 'mac_addr': 'aa:bb:cc:dd:ee:ff', 'ip_addr': '192.168.0.1', 'vni': 500}
        esi = EvpnArbitraryEsi(b'\x00\x00\x00\x00\x00\x00\x00\x00\x00')
        prefix_inst = EvpnMacIPAdvertisementNLRI(route_dist=route_dist, esi=esi, **kwargs)
        next_hop = '10.0.0.1'
        route_family = VRF_RF_L2_EVPN
        route_type = EvpnMacIPAdvertisementNLRI.ROUTE_TYPE_NAME
        tunnel_type = 'vxlan'
        kwargs['esi'] = 0
        tbl_mng = table_manager.TableCoreManager(None, None)
        vrf_table_mock = mock.MagicMock()
        tbl_mng._tables = {(route_dist, route_family): vrf_table_mock}
        tbl_mng.update_vrf_table(route_dist=route_dist, prefix=prefix_str, next_hop=next_hop, route_family=route_family, route_type=route_type, tunnel_type=tunnel_type, **kwargs)
        call_args_list = vrf_table_mock.insert_vrf_path.call_args_list
        self.assertTrue(len(call_args_list) == 1)
        args, kwargs = call_args_list[0]
        self.assertTrue(len(args) == 0)
        self.assertEqual(str(prefix_inst), str(kwargs['nlri']))
        self.assertEqual(next_hop, kwargs['next_hop'])
        self.assertEqual(False, kwargs['gen_lbl'])
        self.assertEqual(tunnel_type, kwargs['tunnel_type'])

    def test_update_vrf_table_ipv4_withdraw(self):
        route_dist = '65000:100'
        ip_network = '192.168.0.0'
        ip_prefix_len = 24
        prefix_str = '%s/%d' % (ip_network, ip_prefix_len)
        prefix_inst = IPAddrPrefix(ip_prefix_len, ip_network)
        next_hop = '10.0.0.1'
        route_family = VRF_RF_IPV4
        route_type = None
        kwargs = {}
        self._test_update_vrf_table(prefix_inst, route_dist, prefix_str, next_hop, route_family, route_type, is_withdraw=True, **kwargs)

    @mock.patch('os_ken.services.protocols.bgp.core_managers.TableCoreManager.__init__', mock.MagicMock(return_value=None))
    def test_update_vrf_table_no_vrf(self):
        route_dist = '65000:100'
        ip_network = '192.168.0.0'
        ip_prefix_len = 24
        prefix_str = '%s/%d' % (ip_network, ip_prefix_len)
        next_hop = '10.0.0.1'
        route_family = VRF_RF_IPV4
        route_type = None
        kwargs = {}
        tbl_mng = table_manager.TableCoreManager(None, None)
        tbl_mng._tables = {}
        self.assertRaises(BgpCoreError, tbl_mng.update_vrf_table, route_dist=route_dist, prefix=prefix_str, next_hop=next_hop, route_family=route_family, route_type=route_type, **kwargs)

    def test_update_vrf_table_invalid_next_hop(self):
        route_dist = '65000:100'
        ip_network = '192.168.0.0'
        ip_prefix_len = 24
        prefix_str = '%s/%d' % (ip_network, ip_prefix_len)
        prefix_inst = IPAddrPrefix(ip_prefix_len, ip_network)
        next_hop = 'xxx.xxx.xxx.xxx'
        route_family = VRF_RF_IPV4
        route_type = None
        kwargs = {}
        self.assertRaises(BgpCoreError, self._test_update_vrf_table, prefix_inst, route_dist, prefix_str, next_hop, route_family, route_type, **kwargs)

    def test_update_vrf_table_invalid_ipv4_prefix(self):
        route_dist = '65000:100'
        ip_network = 'xxx.xxx.xxx.xxx'
        ip_prefix_len = 24
        prefix_str = '%s/%d' % (ip_network, ip_prefix_len)
        prefix_inst = IPAddrPrefix(ip_prefix_len, ip_network)
        next_hop = '10.0.0.1'
        route_family = VRF_RF_IPV4
        route_type = None
        kwargs = {}
        self.assertRaises(BgpCoreError, self._test_update_vrf_table, prefix_inst, route_dist, prefix_str, next_hop, route_family, route_type, **kwargs)

    def test_update_vrf_table_invalid_ipv6_prefix(self):
        route_dist = '65000:100'
        ip_network = 'xxxx::'
        ip_prefix_len = 64
        prefix_str = '%s/%d' % (ip_network, ip_prefix_len)
        prefix_inst = IP6AddrPrefix(ip_prefix_len, ip_network)
        next_hop = 'fe80::0011:aabb:ccdd:eeff'
        route_family = VRF_RF_IPV6
        route_type = None
        kwargs = {}
        self.assertRaises(BgpCoreError, self._test_update_vrf_table, prefix_inst, route_dist, prefix_str, next_hop, route_family, route_type, **kwargs)

    def test_update_vrf_table_invalid_route_family(self):
        route_dist = '65000:100'
        ip_network = '192.168.0.0'
        ip_prefix_len = 24
        prefix_str = '%s/%d' % (ip_network, ip_prefix_len)
        prefix_inst = IPAddrPrefix(ip_prefix_len, ip_network)
        next_hop = '10.0.0.1'
        route_family = 'foobar'
        route_type = None
        kwargs = {}
        self.assertRaises(BgpCoreError, self._test_update_vrf_table, prefix_inst, route_dist, prefix_str, next_hop, route_family, route_type, **kwargs)

    @mock.patch('os_ken.services.protocols.bgp.core_managers.TableCoreManager.__init__', mock.MagicMock(return_value=None))
    @mock.patch('os_ken.services.protocols.bgp.core_managers.TableCoreManager.learn_path')
    def _test_update_global_table(self, learn_path_mock, prefix, next_hop, is_withdraw, expected_next_hop):
        origin = BGPPathAttributeOrigin(BGP_ATTR_ORIGIN_IGP)
        aspath = BGPPathAttributeAsPath([[]])
        pathattrs = OrderedDict()
        pathattrs[BGP_ATTR_TYPE_ORIGIN] = origin
        pathattrs[BGP_ATTR_TYPE_AS_PATH] = aspath
        pathattrs = str(pathattrs)
        tbl_mng = table_manager.TableCoreManager(None, None)
        tbl_mng.update_global_table(prefix=prefix, next_hop=next_hop, is_withdraw=is_withdraw)
        call_args_list = learn_path_mock.call_args_list
        self.assertTrue(len(call_args_list) == 1)
        args, kwargs = call_args_list[0]
        self.assertTrue(len(kwargs) == 0)
        output_path = args[0]
        self.assertEqual(None, output_path.source)
        self.assertEqual(prefix, output_path.nlri.prefix)
        self.assertEqual(pathattrs, str(output_path.pathattr_map))
        self.assertEqual(expected_next_hop, output_path.nexthop)
        self.assertEqual(is_withdraw, output_path.is_withdraw)

    def test_update_global_table_ipv4(self):
        self._test_update_global_table(prefix='192.168.0.0/24', next_hop='10.0.0.1', is_withdraw=False, expected_next_hop='10.0.0.1')

    def test_update_global_table_ipv4_withdraw(self):
        self._test_update_global_table(prefix='192.168.0.0/24', next_hop='10.0.0.1', is_withdraw=True, expected_next_hop='10.0.0.1')

    def test_update_global_table_ipv4_no_next_hop(self):
        self._test_update_global_table(prefix='192.168.0.0/24', next_hop=None, is_withdraw=True, expected_next_hop='0.0.0.0')

    def test_update_global_table_ipv6(self):
        self._test_update_global_table(prefix='fe80::/64', next_hop='fe80::0011:aabb:ccdd:eeff', is_withdraw=False, expected_next_hop='fe80::0011:aabb:ccdd:eeff')

    def test_update_global_table_ipv6_withdraw(self):
        self._test_update_global_table(prefix='fe80::/64', next_hop='fe80::0011:aabb:ccdd:eeff', is_withdraw=True, expected_next_hop='fe80::0011:aabb:ccdd:eeff')

    def test_update_global_table_ipv6_no_next_hop(self):
        self._test_update_global_table(prefix='fe80::/64', next_hop=None, is_withdraw=True, expected_next_hop='::')

    @mock.patch('os_ken.services.protocols.bgp.core_managers.TableCoreManager.__init__', mock.MagicMock(return_value=None))
    def _test_update_flowspec_vrf_table(self, flowspec_family, route_family, route_dist, rules, prefix, is_withdraw, actions=None):
        tbl_mng = table_manager.TableCoreManager(None, None)
        vrf_table_mock = mock.MagicMock()
        tbl_mng._tables = {(route_dist, route_family): vrf_table_mock}
        tbl_mng.update_flowspec_vrf_table(flowspec_family=flowspec_family, route_dist=route_dist, rules=rules, actions=actions, is_withdraw=is_withdraw)
        call_args_list = vrf_table_mock.insert_vrffs_path.call_args_list
        self.assertTrue(len(call_args_list) == 1)
        args, kwargs = call_args_list[0]
        self.assertTrue(len(args) == 0)
        self.assertEqual(prefix, kwargs['nlri'].prefix)
        self.assertEqual(is_withdraw, kwargs['is_withdraw'])

    def test_update_flowspec_vrf_table_vpnv4(self):
        flowspec_family = 'vpnv4fs'
        route_family = 'ipv4fs'
        route_dist = '65001:100'
        rules = {'dst_prefix': '10.70.1.0/24'}
        actions = {'traffic_rate': {'as_number': 0, 'rate_info': 100.0}}
        prefix = 'ipv4fs(dst_prefix:10.70.1.0/24)'
        self._test_update_flowspec_vrf_table(flowspec_family=flowspec_family, route_family=route_family, route_dist=route_dist, rules=rules, prefix=prefix, is_withdraw=False, actions=actions)

    def test_update_flowspec_vrf_table_vpnv4_without_actions(self):
        flowspec_family = 'vpnv4fs'
        route_family = 'ipv4fs'
        route_dist = '65001:100'
        rules = {'dst_prefix': '10.70.1.0/24'}
        prefix = 'ipv4fs(dst_prefix:10.70.1.0/24)'
        self._test_update_flowspec_vrf_table(flowspec_family=flowspec_family, route_family=route_family, route_dist=route_dist, rules=rules, prefix=prefix, is_withdraw=False)

    def test_update_flowspec_vrf_table_vpnv4_invalid_actions(self):
        flowspec_family = 'vpnv4fs'
        route_family = 'ipv4fs'
        route_dist = '65001:100'
        rules = {'dst_prefix': '10.70.1.0/24'}
        actions = {'invalid_actions': {'invalid_param': 10}}
        prefix = 'ipv4fs(dst_prefix:10.70.1.0/24)'
        self.assertRaises(BgpCoreError, self._test_update_flowspec_vrf_table, flowspec_family=flowspec_family, route_family=route_family, route_dist=route_dist, rules=rules, prefix=prefix, is_withdraw=False, actions=actions)

    def test_update_flowspec_vrf_table_vpnv4_invalid_flowspec_family(self):
        flowspec_family = 'invalid'
        route_family = 'ipv4fs'
        route_dist = '65001:100'
        rules = {'dst_prefix': '10.70.1.0/24'}
        prefix = 'ipv4fs(dst_prefix:10.70.1.0/24)'
        self.assertRaises(BgpCoreError, self._test_update_flowspec_vrf_table, flowspec_family=flowspec_family, route_family=route_family, route_dist=route_dist, rules=rules, prefix=prefix, is_withdraw=False)

    def test_update_flowspec_vrf_table_vpnv4_invalid_route_family(self):
        flowspec_family = 'vpnv4fs'
        route_family = 'invalid'
        route_dist = '65001:100'
        rules = {'dst_prefix': '10.70.1.0/24'}
        prefix = 'ipv4fs(dst_prefix:10.70.1.0/24)'
        self.assertRaises(BgpCoreError, self._test_update_flowspec_vrf_table, flowspec_family=flowspec_family, route_family=route_family, route_dist=route_dist, rules=rules, prefix=prefix, is_withdraw=False)

    @mock.patch('os_ken.services.protocols.bgp.core_managers.TableCoreManager.__init__', mock.MagicMock(return_value=None))
    @mock.patch('os_ken.services.protocols.bgp.core_managers.TableCoreManager.learn_path')
    def _test_update_flowspec_global_table(self, learn_path_mock, flowspec_family, rules, prefix, is_withdraw, actions=None):
        tbl_mng = table_manager.TableCoreManager(None, None)
        tbl_mng.update_flowspec_global_table(flowspec_family=flowspec_family, rules=rules, actions=actions, is_withdraw=is_withdraw)
        call_args_list = learn_path_mock.call_args_list
        self.assertTrue(len(call_args_list) == 1)
        args, kwargs = call_args_list[0]
        self.assertTrue(len(kwargs) == 0)
        output_path = args[0]
        self.assertEqual(None, output_path.source)
        self.assertEqual(prefix, output_path.nlri.prefix)
        self.assertEqual(None, output_path.nexthop)
        self.assertEqual(is_withdraw, output_path.is_withdraw)

    def test_update_flowspec_global_table_ipv4(self):
        flowspec_family = 'ipv4fs'
        rules = {'dst_prefix': '10.60.1.0/24'}
        actions = {'traffic_rate': {'as_number': 0, 'rate_info': 100.0}}
        prefix = 'ipv4fs(dst_prefix:10.60.1.0/24)'
        self._test_update_flowspec_global_table(flowspec_family=flowspec_family, rules=rules, prefix=prefix, is_withdraw=False, actions=actions)

    def test_update_flowspec_global_table_ipv4_without_actions(self):
        flowspec_family = 'ipv4fs'
        rules = {'dst_prefix': '10.60.1.0/24'}
        prefix = 'ipv4fs(dst_prefix:10.60.1.0/24)'
        self._test_update_flowspec_global_table(flowspec_family=flowspec_family, rules=rules, prefix=prefix, is_withdraw=False)

    def test_update_flowspec_global_table_ipv4_invalid_actions(self):
        flowspec_family = 'ipv4fs'
        rules = {'dst_prefix': '10.60.1.0/24'}
        actions = {'invalid_actions': {'invalid_param': 10}}
        prefix = 'ipv4fs(dst_prefix:10.60.1.0/24)'
        self.assertRaises(BgpCoreError, self._test_update_flowspec_global_table, flowspec_family=flowspec_family, rules=rules, prefix=prefix, is_withdraw=False, actions=actions)

    def test_update_flowspec_global_table_ipv4_invalid_flowspec_family(self):
        flowspec_family = 'invalid'
        rules = {'dst_prefix': '10.60.1.0/24'}
        actions = {'traffic_rate': {'as_number': 0, 'rate_info': 100.0}}
        prefix = 'ipv4fs(dst_prefix:10.60.1.0/24)'
        self.assertRaises(BgpCoreError, self._test_update_flowspec_global_table, flowspec_family=flowspec_family, rules=rules, prefix=prefix, is_withdraw=False, actions=actions)

    def test_update_flowspec_global_table_ipv6(self):
        flowspec_family = 'ipv6fs'
        rules = {'dst_prefix': '2001::3/128/32'}
        actions = {'traffic_rate': {'as_number': 0, 'rate_info': 100.0}}
        prefix = 'ipv6fs(dst_prefix:2001::3/128/32)'
        self._test_update_flowspec_global_table(flowspec_family=flowspec_family, rules=rules, prefix=prefix, is_withdraw=False, actions=actions)

    def test_update_flowspec_global_table_ipv6_without_actions(self):
        flowspec_family = 'ipv6fs'
        rules = {'dst_prefix': '2001::3/128/32'}
        prefix = 'ipv6fs(dst_prefix:2001::3/128/32)'
        self._test_update_flowspec_global_table(flowspec_family=flowspec_family, rules=rules, prefix=prefix, is_withdraw=False)

    def test_update_flowspec_global_table_ipv6_invalid_actions(self):
        flowspec_family = 'ipv6fs'
        rules = {'dst_prefix': '2001::3/128/32'}
        actions = {'invalid_actions': {'invalid_param': 10}}
        prefix = 'ipv4fs(dst_prefix:2001::3/128/32)'
        self.assertRaises(BgpCoreError, self._test_update_flowspec_global_table, flowspec_family=flowspec_family, rules=rules, prefix=prefix, is_withdraw=False, actions=actions)

    def test_update_flowspec_global_table_ipv6_invalid_flowspec_family(self):
        flowspec_family = 'invalid'
        rules = {'dst_prefix': '2001::3/128/32'}
        actions = {'traffic_rate': {'as_number': 0, 'rate_info': 100.0}}
        prefix = 'ipv4fs(dst_prefix:2001::3/128/32)'
        self.assertRaises(BgpCoreError, self._test_update_flowspec_global_table, flowspec_family=flowspec_family, rules=rules, prefix=prefix, is_withdraw=False, actions=actions)

    def test_update_flowspec_vrf_table_vpnv6(self):
        flowspec_family = 'vpnv6fs'
        route_family = 'ipv6fs'
        route_dist = '65001:100'
        rules = {'dst_prefix': '2001::3/128/32'}
        actions = {'traffic_rate': {'as_number': 0, 'rate_info': 100.0}}
        prefix = 'ipv6fs(dst_prefix:2001::3/128/32)'
        self._test_update_flowspec_vrf_table(flowspec_family=flowspec_family, route_family=route_family, route_dist=route_dist, rules=rules, prefix=prefix, is_withdraw=False, actions=actions)

    def test_update_flowspec_vrf_table_vpnv6_without_actions(self):
        flowspec_family = 'vpnv6fs'
        route_family = 'ipv6fs'
        route_dist = '65001:100'
        rules = {'dst_prefix': '2001::3/128/32'}
        prefix = 'ipv6fs(dst_prefix:2001::3/128/32)'
        self._test_update_flowspec_vrf_table(flowspec_family=flowspec_family, route_family=route_family, route_dist=route_dist, rules=rules, prefix=prefix, is_withdraw=False)

    def test_update_flowspec_vrf_table_vpnv6_invalid_actions(self):
        flowspec_family = 'vpnv6fs'
        route_family = 'ipv6fs'
        route_dist = '65001:100'
        rules = {'dst_prefix': '2001::3/128/32'}
        actions = {'invalid_actions': {'invalid_param': 10}}
        prefix = 'ipv6fs(dst_prefix:2001::3/128/32)'
        self.assertRaises(BgpCoreError, self._test_update_flowspec_vrf_table, flowspec_family=flowspec_family, route_family=route_family, route_dist=route_dist, rules=rules, prefix=prefix, is_withdraw=False, actions=actions)

    def test_update_flowspec_vrf_table_vpnv6_invalid_route_family(self):
        flowspec_family = 'vpnv6fs'
        route_family = 'invalid'
        route_dist = '65001:100'
        rules = {'dst_prefix': '2001::3/128/32'}
        prefix = 'ipv4fs(dst_prefix:2001::3/128/32)'
        self.assertRaises(BgpCoreError, self._test_update_flowspec_vrf_table, flowspec_family=flowspec_family, route_family=route_family, route_dist=route_dist, rules=rules, prefix=prefix, is_withdraw=False)

    def test_update_flowspec_vrf_table_l2vpn(self):
        flowspec_family = 'l2vpnfs'
        route_family = 'l2vpnfs'
        route_dist = '65001:100'
        rules = {'dst_mac': '12:34:56:78:9a:bc'}
        actions = {'traffic_rate': {'as_number': 0, 'rate_info': 100.0}}
        prefix = 'l2vpnfs(dst_mac:12:34:56:78:9a:bc)'
        self._test_update_flowspec_vrf_table(flowspec_family=flowspec_family, route_family=route_family, route_dist=route_dist, rules=rules, prefix=prefix, is_withdraw=False, actions=actions)

    def test_update_flowspec_vrf_table_l2vpn_without_actions(self):
        flowspec_family = 'l2vpnfs'
        route_family = 'l2vpnfs'
        route_dist = '65001:100'
        rules = {'dst_mac': '12:34:56:78:9a:bc'}
        prefix = 'l2vpnfs(dst_mac:12:34:56:78:9a:bc)'
        self._test_update_flowspec_vrf_table(flowspec_family=flowspec_family, route_family=route_family, route_dist=route_dist, rules=rules, prefix=prefix, is_withdraw=False)

    def test_update_flowspec_vrf_table_l2vpn_invalid_actions(self):
        flowspec_family = 'l2vpnfs'
        route_family = 'l2vpnfs'
        route_dist = '65001:100'
        rules = {'dst_mac': '12:34:56:78:9a:bc'}
        actions = {'invalid_actions': {'invalid_param': 10}}
        prefix = 'l2vpnfs(dst_mac:12:34:56:78:9a:bc)'
        self.assertRaises(BgpCoreError, self._test_update_flowspec_vrf_table, flowspec_family=flowspec_family, route_family=route_family, route_dist=route_dist, rules=rules, prefix=prefix, is_withdraw=False, actions=actions)

    def test_update_flowspec_vrf_table_l2vpn_invalid_route_family(self):
        flowspec_family = 'l2vpnfs'
        route_family = 'invalid'
        route_dist = '65001:100'
        rules = {'dst_mac': '12:34:56:78:9a:bc'}
        prefix = 'l2vpnfs(dst_mac:12:34:56:78:9a:bc)'
        self.assertRaises(BgpCoreError, self._test_update_flowspec_vrf_table, flowspec_family=flowspec_family, route_family=route_family, route_dist=route_dist, rules=rules, prefix=prefix, is_withdraw=False)