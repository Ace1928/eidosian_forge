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