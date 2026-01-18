import logging
from os_ken.lib.packet.bgp import RF_L2_EVPN
from os_ken.lib.packet.bgp import EvpnNLRI
from os_ken.services.protocols.bgp.info_base.evpn import EvpnPath
from os_ken.services.protocols.bgp.info_base.vrf import VrfDest
from os_ken.services.protocols.bgp.info_base.vrf import VrfNlriImportMap
from os_ken.services.protocols.bgp.info_base.vrf import VrfPath
from os_ken.services.protocols.bgp.info_base.vrf import VrfTable
class VrfEvpnDest(VrfDest):
    """Destination for EVPN VRFs."""
    ROUTE_FAMILY = RF_L2_EVPN