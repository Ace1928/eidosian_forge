import logging
from os_ken.lib.packet.bgp import IP6AddrPrefix
from os_ken.lib.packet.bgp import RF_IPv6_VPN
from os_ken.services.protocols.bgp.info_base.vpn import VpnDest
from os_ken.services.protocols.bgp.info_base.vpn import VpnPath
from os_ken.services.protocols.bgp.info_base.vpn import VpnTable
class Vpnv6Dest(VpnDest):
    """VPNv6 destination

    Stores IPv6 paths.
    """
    ROUTE_FAMILY = RF_IPv6_VPN