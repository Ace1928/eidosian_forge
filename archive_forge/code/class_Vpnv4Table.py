import logging
from os_ken.lib.packet.bgp import IPAddrPrefix
from os_ken.lib.packet.bgp import RF_IPv4_VPN
from os_ken.services.protocols.bgp.info_base.vpn import VpnDest
from os_ken.services.protocols.bgp.info_base.vpn import VpnPath
from os_ken.services.protocols.bgp.info_base.vpn import VpnTable
class Vpnv4Table(VpnTable):
    """Global table to store VPNv4 routing information.

    Uses `Vpnv4Dest` to store destination information for each known vpnv4
    paths.
    """
    ROUTE_FAMILY = RF_IPv4_VPN
    VPN_DEST_CLASS = Vpnv4Dest