import logging
from os_ken.lib.packet.bgp import RF_IPv4_FLOWSPEC
from os_ken.lib.packet.bgp import RF_VPNv4_FLOWSPEC
from os_ken.lib.packet.bgp import FlowSpecIPv4NLRI
from os_ken.lib.packet.bgp import FlowSpecVPNv4NLRI
from os_ken.services.protocols.bgp.info_base.vpnv4fs import VPNv4FlowSpecPath
from os_ken.services.protocols.bgp.info_base.vrffs import VRFFlowSpecDest
from os_ken.services.protocols.bgp.info_base.vrffs import VRFFlowSpecPath
from os_ken.services.protocols.bgp.info_base.vrffs import VRFFlowSpecTable
class Vrf4FlowSpecTable(VRFFlowSpecTable):
    """Virtual Routing and Forwarding information base
    for IPv4 Flow Specification.
    """
    ROUTE_FAMILY = RF_IPv4_FLOWSPEC
    VPN_ROUTE_FAMILY = RF_VPNv4_FLOWSPEC
    NLRI_CLASS = FlowSpecIPv4NLRI
    VRF_PATH_CLASS = Vrf4FlowSpecPath
    VRF_DEST_CLASS = Vrf4FlowSpecDest