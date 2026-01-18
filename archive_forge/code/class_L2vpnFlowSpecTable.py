import logging
from os_ken.lib.packet.bgp import RF_L2VPN_FLOWSPEC
from os_ken.lib.packet.bgp import FlowSpecL2VPNNLRI
from os_ken.services.protocols.bgp.info_base.l2vpnfs import L2VPNFlowSpecPath
from os_ken.services.protocols.bgp.info_base.vrffs import VRFFlowSpecDest
from os_ken.services.protocols.bgp.info_base.vrffs import VRFFlowSpecPath
from os_ken.services.protocols.bgp.info_base.vrffs import VRFFlowSpecTable
class L2vpnFlowSpecTable(VRFFlowSpecTable):
    """Virtual Routing and Forwarding information base
    for L2VPN Flow Specification.
    """
    ROUTE_FAMILY = RF_L2VPN_FLOWSPEC
    VPN_ROUTE_FAMILY = RF_L2VPN_FLOWSPEC
    NLRI_CLASS = FlowSpecL2VPNNLRI
    VRF_PATH_CLASS = L2vpnFlowSpecPath
    VRF_DEST_CLASS = L2vpnFlowSpecDest