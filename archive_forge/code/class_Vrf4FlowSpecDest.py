import logging
from os_ken.lib.packet.bgp import RF_IPv4_FLOWSPEC
from os_ken.lib.packet.bgp import RF_VPNv4_FLOWSPEC
from os_ken.lib.packet.bgp import FlowSpecIPv4NLRI
from os_ken.lib.packet.bgp import FlowSpecVPNv4NLRI
from os_ken.services.protocols.bgp.info_base.vpnv4fs import VPNv4FlowSpecPath
from os_ken.services.protocols.bgp.info_base.vrffs import VRFFlowSpecDest
from os_ken.services.protocols.bgp.info_base.vrffs import VRFFlowSpecPath
from os_ken.services.protocols.bgp.info_base.vrffs import VRFFlowSpecTable
class Vrf4FlowSpecDest(VRFFlowSpecDest):
    ROUTE_FAMILY = RF_IPv4_FLOWSPEC