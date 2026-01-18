import logging
from os_ken.lib.packet.bgp import IP6AddrPrefix
from os_ken.lib.packet.bgp import RF_IPv6_VPN
from os_ken.services.protocols.bgp.info_base.vpn import VpnDest
from os_ken.services.protocols.bgp.info_base.vpn import VpnPath
from os_ken.services.protocols.bgp.info_base.vpn import VpnTable
class Vpnv6Table(VpnTable):
    """Global table to store VPNv6 routing information

    Uses `Vpnv6Dest` to store destination information for each known vpnv6
    paths.
    """
    ROUTE_FAMILY = RF_IPv6_VPN
    VPN_DEST_CLASS = Vpnv6Dest