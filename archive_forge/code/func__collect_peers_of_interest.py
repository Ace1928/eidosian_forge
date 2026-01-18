import logging
import netaddr
from os_ken.services.protocols.bgp.base import SUPPORTED_GLOBAL_RF
from os_ken.services.protocols.bgp.model import OutgoingRoute
from os_ken.services.protocols.bgp.peer import Peer
from os_ken.lib.packet.bgp import BGPPathAttributeCommunities
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_MULTI_EXIT_DISC
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_COMMUNITIES
from os_ken.lib.packet.bgp import RF_RTC_UC
from os_ken.lib.packet.bgp import RouteTargetMembershipNLRI
from os_ken.services.protocols.bgp.utils.bgp \
def _collect_peers_of_interest(self, new_best_path):
    """Collect all peers that qualify for sharing a path with given RTs.
        """
    path_rts = new_best_path.get_rts()
    qualified_peers = set(self._peers.values())
    qualified_peers = self._rt_manager.filter_by_origin_as(new_best_path, qualified_peers)
    if path_rts:
        path_rts.append(RouteTargetMembershipNLRI.DEFAULT_RT)
        qualified_peers = set(self._get_non_rtc_peers())
        peer_to_rtfilter_map = self._peer_to_rtfilter_map
        for peer, rt_filter in peer_to_rtfilter_map.items():
            if peer is None:
                continue
            if rt_filter is None:
                qualified_peers.add(peer)
            elif rt_filter.intersection(path_rts):
                qualified_peers.add(peer)
    return qualified_peers