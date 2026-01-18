import logging
from time import gmtime
class FlexinetOutgoingRoute(object):
    """Holds state about a route that is queued for being sent to a given sink.

    In this case the sink is flexinet peer and this route information is from
    a VRF which holds Ipv4(v6) NLRIs.
    """
    __slots__ = ('_path', 'sink', 'next_outgoing_route', 'prev_outgoing_route', 'next_sink_out_route', 'prev_sink_out_route', '_route_dist')

    def __init__(self, path, route_dist):
        from os_ken.services.protocols.bgp.info_base.vrf4 import Vrf4Path
        from os_ken.services.protocols.bgp.info_base.vrf6 import Vrf6Path
        from os_ken.services.protocols.bgp.info_base.vrfevpn import VrfEvpnPath
        from os_ken.services.protocols.bgp.info_base.vrf4fs import Vrf4FlowSpecPath
        from os_ken.services.protocols.bgp.info_base.vrf6fs import Vrf6FlowSpecPath
        from os_ken.services.protocols.bgp.info_base.vrfl2vpnfs import L2vpnFlowSpecPath
        assert path.route_family in (Vrf4Path.ROUTE_FAMILY, Vrf6Path.ROUTE_FAMILY, VrfEvpnPath.ROUTE_FAMILY, Vrf4FlowSpecPath.ROUTE_FAMILY, Vrf6FlowSpecPath.ROUTE_FAMILY, L2vpnFlowSpecPath.ROUTE_FAMILY)
        self.sink = None
        self._path = path
        self._route_dist = route_dist

    @property
    def path(self):
        return self._path

    @property
    def route_dist(self):
        return self._route_dist

    def __str__(self):
        return 'FlexinetOutgoingRoute(path: %s, route_dist: %s)' % (self.path, self.route_dist)