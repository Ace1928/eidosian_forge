import logging
import netaddr
import socket
from os_ken.lib.packet.bgp import BGP_ERROR_CEASE
from os_ken.lib.packet.bgp import BGP_ERROR_SUB_CONNECTION_RESET
from os_ken.lib.packet.bgp import BGP_ERROR_SUB_CONNECTION_COLLISION_RESOLUTION
from os_ken.lib.packet.bgp import RF_RTC_UC
from os_ken.lib.packet.bgp import BGP_ATTR_ORIGIN_INCOMPLETE
from os_ken.services.protocols.bgp.base import Activity
from os_ken.services.protocols.bgp.base import add_bgp_error_metadata
from os_ken.services.protocols.bgp.base import BGPSException
from os_ken.services.protocols.bgp.base import CORE_ERROR_CODE
from os_ken.services.protocols.bgp.constants import STD_BGP_SERVER_PORT_NUM
from os_ken.services.protocols.bgp import core_managers
from os_ken.services.protocols.bgp.model import FlexinetOutgoingRoute
from os_ken.services.protocols.bgp.protocol import Factory
from os_ken.services.protocols.bgp.signals.emit import BgpSignalBus
from os_ken.services.protocols.bgp.speaker import BgpProtocol
from os_ken.services.protocols.bgp.utils.rtfilter import RouteTargetManager
from os_ken.services.protocols.bgp.rtconf.neighbors import CONNECT_MODE_ACTIVE
from os_ken.services.protocols.bgp.utils import stats
from os_ken.services.protocols.bgp.bmp import BMPClient
from os_ken.lib import sockopt
from os_ken.lib import ip
def _on_update_rt_filter(self, peer, new_rts, old_rts):
    """Handles update of peer RT filter.

        Parameters:
            - `peer`: (Peer) whose RT filter has changed.
            - `new_rts`: (set) of new RTs that peer is interested in.
            - `old_rts`: (set) of RTs that peers is no longer interested in.
        """
    for table in self._table_manager._global_tables.values():
        if table.route_family == RF_RTC_UC:
            continue
        self._spawn('rt_filter_chg_%s' % peer, self._rt_mgr.on_rt_filter_chg_sync_peer, peer, new_rts, old_rts, table)
        LOG.debug('RT Filter change handler launched for route_family %s', table.route_family)