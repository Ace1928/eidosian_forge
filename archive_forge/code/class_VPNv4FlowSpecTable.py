import logging
from os_ken.lib.packet.bgp import FlowSpecVPNv4NLRI
from os_ken.lib.packet.bgp import RF_VPNv4_FLOWSPEC
from os_ken.services.protocols.bgp.info_base.vpn import VpnDest
from os_ken.services.protocols.bgp.info_base.vpn import VpnPath
from os_ken.services.protocols.bgp.info_base.vpn import VpnTable
class VPNv4FlowSpecTable(VpnTable):
    """Global table to store VPNv4 Flow Specification routing information.

    Uses `VPNv4FlowSpecDest` to store destination information for each known
    Flow Specification paths.
    """
    ROUTE_FAMILY = RF_VPNv4_FLOWSPEC
    VPN_DEST_CLASS = VPNv4FlowSpecDest