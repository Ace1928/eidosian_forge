import abc
import logging
from os_ken.lib.packet.bgp import RF_L2_EVPN
from os_ken.services.protocols.bgp.info_base.base import Destination
from os_ken.services.protocols.bgp.info_base.base import NonVrfPathProcessingMixin
from os_ken.services.protocols.bgp.info_base.base import Path
from os_ken.services.protocols.bgp.info_base.base import Table
class VpnTable(Table):
    """Global table to store VPNv4 routing information.

    Uses `VpnvXDest` to store destination information for each known vpnvX
    paths.
    """
    ROUTE_FAMILY = None
    VPN_DEST_CLASS = None

    def __init__(self, core_service, signal_bus):
        super(VpnTable, self).__init__(None, core_service, signal_bus)

    def _table_key(self, vpn_nlri):
        """Return a key that will uniquely identify this vpnvX NLRI inside
        this table.
        """
        return vpn_nlri.route_dist + ':' + vpn_nlri.prefix

    def _create_dest(self, nlri):
        return self.VPN_DEST_CLASS(self, nlri)

    def __str__(self):
        return '%s(scope_id: %s, rf: %s)' % (self.__class__.__name__, self.scope_id, self.route_family)