import logging
from os_ken.lib.packet.bgp import FlowSpecVPNv4NLRI
from os_ken.lib.packet.bgp import RF_VPNv4_FLOWSPEC
from os_ken.services.protocols.bgp.info_base.vpn import VpnDest
from os_ken.services.protocols.bgp.info_base.vpn import VpnPath
from os_ken.services.protocols.bgp.info_base.vpn import VpnTable
class VPNv4FlowSpecPath(VpnPath):
    """Represents a way of reaching an VPNv4 Flow Specification destination."""
    ROUTE_FAMILY = RF_VPNv4_FLOWSPEC
    VRF_PATH_CLASS = None
    NLRI_CLASS = FlowSpecVPNv4NLRI

    def __init__(self, *args, **kwargs):
        kwargs['nexthop'] = '0.0.0.0'
        super(VPNv4FlowSpecPath, self).__init__(*args, **kwargs)
        from os_ken.services.protocols.bgp.info_base.vrf4fs import Vrf4FlowSpecPath
        self.VRF_PATH_CLASS = Vrf4FlowSpecPath
        self._nexthop = None