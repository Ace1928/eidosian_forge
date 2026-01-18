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