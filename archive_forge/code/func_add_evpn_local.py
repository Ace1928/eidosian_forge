import logging
from os_ken.lib.packet.bgp import EvpnEsi
from os_ken.lib.packet.bgp import EvpnNLRI
from os_ken.lib.packet.bgp import EvpnEthernetAutoDiscoveryNLRI
from os_ken.lib.packet.bgp import EvpnMacIPAdvertisementNLRI
from os_ken.lib.packet.bgp import EvpnInclusiveMulticastEthernetTagNLRI
from os_ken.lib.packet.bgp import EvpnEthernetSegmentNLRI
from os_ken.lib.packet.bgp import EvpnIpPrefixNLRI
from os_ken.lib.packet.bgp import BGPPathAttributePmsiTunnel
from os_ken.lib.packet.bgp import FlowSpecIPv4NLRI
from os_ken.lib.packet.bgp import FlowSpecIPv6NLRI
from os_ken.lib.packet.bgp import FlowSpecVPNv4NLRI
from os_ken.lib.packet.bgp import FlowSpecVPNv6NLRI
from os_ken.lib.packet.bgp import FlowSpecL2VPNNLRI
from os_ken.lib.packet.bgp import BGPFlowSpecTrafficRateCommunity
from os_ken.lib.packet.bgp import BGPFlowSpecTrafficActionCommunity
from os_ken.lib.packet.bgp import BGPFlowSpecRedirectCommunity
from os_ken.lib.packet.bgp import BGPFlowSpecTrafficMarkingCommunity
from os_ken.lib.packet.bgp import BGPFlowSpecVlanActionCommunity
from os_ken.lib.packet.bgp import BGPFlowSpecTPIDActionCommunity
from os_ken.services.protocols.bgp.api.base import EVPN_ROUTE_TYPE
from os_ken.services.protocols.bgp.api.base import EVPN_ESI
from os_ken.services.protocols.bgp.api.base import EVPN_ETHERNET_TAG_ID
from os_ken.services.protocols.bgp.api.base import REDUNDANCY_MODE
from os_ken.services.protocols.bgp.api.base import MAC_ADDR
from os_ken.services.protocols.bgp.api.base import IP_ADDR
from os_ken.services.protocols.bgp.api.base import IP_PREFIX
from os_ken.services.protocols.bgp.api.base import GW_IP_ADDR
from os_ken.services.protocols.bgp.api.base import MPLS_LABELS
from os_ken.services.protocols.bgp.api.base import NEXT_HOP
from os_ken.services.protocols.bgp.api.base import PREFIX
from os_ken.services.protocols.bgp.api.base import RegisterWithArgChecks
from os_ken.services.protocols.bgp.api.base import ROUTE_DISTINGUISHER
from os_ken.services.protocols.bgp.api.base import VPN_LABEL
from os_ken.services.protocols.bgp.api.base import EVPN_VNI
from os_ken.services.protocols.bgp.api.base import TUNNEL_TYPE
from os_ken.services.protocols.bgp.api.base import PMSI_TUNNEL_TYPE
from os_ken.services.protocols.bgp.api.base import MAC_MOBILITY
from os_ken.services.protocols.bgp.api.base import TUNNEL_ENDPOINT_IP
from os_ken.services.protocols.bgp.api.base import FLOWSPEC_FAMILY
from os_ken.services.protocols.bgp.api.base import FLOWSPEC_RULES
from os_ken.services.protocols.bgp.api.base import FLOWSPEC_ACTIONS
from os_ken.services.protocols.bgp.base import add_bgp_error_metadata
from os_ken.services.protocols.bgp.base import PREFIX_ERROR_CODE
from os_ken.services.protocols.bgp.base import validate
from os_ken.services.protocols.bgp.core import BgpCoreError
from os_ken.services.protocols.bgp.core_manager import CORE_MANAGER
from os_ken.services.protocols.bgp.rtconf.base import ConfigValueError
from os_ken.services.protocols.bgp.rtconf.base import RuntimeConfigError
from os_ken.services.protocols.bgp.rtconf.vrfs import VRF_RF
from os_ken.services.protocols.bgp.rtconf.vrfs import VRF_RF_IPV4
from os_ken.services.protocols.bgp.rtconf.vrfs import VRF_RF_L2_EVPN
from os_ken.services.protocols.bgp.utils import validation
@RegisterWithArgChecks(name='evpn_prefix.add_local', req_args=[EVPN_ROUTE_TYPE, ROUTE_DISTINGUISHER, NEXT_HOP], opt_args=[EVPN_ESI, EVPN_ETHERNET_TAG_ID, REDUNDANCY_MODE, MAC_ADDR, IP_ADDR, IP_PREFIX, GW_IP_ADDR, EVPN_VNI, TUNNEL_TYPE, PMSI_TUNNEL_TYPE, TUNNEL_ENDPOINT_IP, MAC_MOBILITY])
def add_evpn_local(route_type, route_dist, next_hop, **kwargs):
    """Adds EVPN route from VRF identified by *route_dist*.
    """
    if route_type in [EVPN_ETH_AUTO_DISCOVERY, EVPN_ETH_SEGMENT] and kwargs['esi'] == 0:
        raise ConfigValueError(conf_name=EVPN_ESI, conf_value=kwargs['esi'])
    try:
        tm = CORE_MANAGER.get_core_service().table_manager
        label = tm.update_vrf_table(route_dist, next_hop=next_hop, route_family=VRF_RF_L2_EVPN, route_type=route_type, **kwargs)
        if label:
            label = label[0]
        return [{EVPN_ROUTE_TYPE: route_type, ROUTE_DISTINGUISHER: route_dist, VRF_RF: VRF_RF_L2_EVPN, VPN_LABEL: label}.update(kwargs)]
    except BgpCoreError as e:
        raise PrefixError(desc=e)