import logging
from os_ken.lib.packet.bgp import FlowSpecIPv4NLRI
from os_ken.lib.packet.bgp import RF_IPv4_FLOWSPEC
from os_ken.services.protocols.bgp.info_base.base import Path
from os_ken.services.protocols.bgp.info_base.base import Table
from os_ken.services.protocols.bgp.info_base.base import Destination
from os_ken.services.protocols.bgp.info_base.base import NonVrfPathProcessingMixin
class IPv4FlowSpecTable(Table):
    """Global table to store IPv4 Flow Specification routing information.

    Uses `FlowSpecIpv4Dest` to store destination information for each known
    Flow Specification paths.
    """
    ROUTE_FAMILY = RF_IPv4_FLOWSPEC
    VPN_DEST_CLASS = IPv4FlowSpecDest

    def __init__(self, core_service, signal_bus):
        super(IPv4FlowSpecTable, self).__init__(None, core_service, signal_bus)

    def _table_key(self, nlri):
        """Return a key that will uniquely identify this NLRI inside
        this table.
        """
        return nlri.prefix

    def _create_dest(self, nlri):
        return self.VPN_DEST_CLASS(self, nlri)

    def __str__(self):
        return '%s(scope_id: %s, rf: %s)' % (self.__class__.__name__, self.scope_id, self.route_family)