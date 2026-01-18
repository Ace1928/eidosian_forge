import logging
from os_ken.lib.packet.bgp import FlowSpecVPNv6NLRI
from os_ken.lib.packet.bgp import RF_VPNv6_FLOWSPEC
from os_ken.services.protocols.bgp.info_base.vpn import VpnDest
from os_ken.services.protocols.bgp.info_base.vpn import VpnPath
from os_ken.services.protocols.bgp.info_base.vpn import VpnTable
class VPNv6FlowSpecTable(VpnTable):
    """Global table to store VPNv6 Flow Specification routing information.

    Uses `VPNv6FlowSpecDest` to store destination information for each known
    Flow Specification paths.
    """
    ROUTE_FAMILY = RF_VPNv6_FLOWSPEC
    VPN_DEST_CLASS = VPNv6FlowSpecDest