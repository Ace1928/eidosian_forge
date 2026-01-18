import logging
from os_ken.lib.packet.bgp import FlowSpecIPv4NLRI
from os_ken.lib.packet.bgp import RF_IPv4_FLOWSPEC
from os_ken.services.protocols.bgp.info_base.base import Path
from os_ken.services.protocols.bgp.info_base.base import Table
from os_ken.services.protocols.bgp.info_base.base import Destination
from os_ken.services.protocols.bgp.info_base.base import NonVrfPathProcessingMixin
class IPv4FlowSpecPath(Path):
    """Represents a way of reaching an IPv4 Flow Specification destination."""
    ROUTE_FAMILY = RF_IPv4_FLOWSPEC
    VRF_PATH_CLASS = None
    NLRI_CLASS = FlowSpecIPv4NLRI

    def __init__(self, *args, **kwargs):
        kwargs['nexthop'] = '0.0.0.0'
        super(IPv4FlowSpecPath, self).__init__(*args, **kwargs)
        from os_ken.services.protocols.bgp.info_base.vrf4fs import Vrf4FlowSpecPath
        self.VRF_PATH_CLASS = Vrf4FlowSpecPath
        self._nexthop = None