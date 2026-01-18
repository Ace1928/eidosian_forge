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