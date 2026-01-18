import logging
from os_ken.lib.packet.bgp import RF_RTC_UC
from os_ken.services.protocols.bgp.info_base.base import Destination
from os_ken.services.protocols.bgp.info_base.base import NonVrfPathProcessingMixin
from os_ken.services.protocols.bgp.info_base.base import Path
from os_ken.services.protocols.bgp.info_base.base import Table
class RtcTable(Table):
    """Global table to store RT membership information.

    Uses `RtDest` to store destination information for each known RT NLRI path.
    """
    ROUTE_FAMILY = RF_RTC_UC

    def __init__(self, core_service, signal_bus):
        Table.__init__(self, None, core_service, signal_bus)

    def _table_key(self, rtc_nlri):
        """Return a key that will uniquely identify this RT NLRI inside
        this table.
        """
        return str(rtc_nlri.origin_as) + ':' + rtc_nlri.route_target

    def _create_dest(self, nlri):
        return RtcDest(self, nlri)

    def __str__(self):
        return 'RtcTable(scope_id: %s, rf: %s)' % (self.scope_id, self.route_family)