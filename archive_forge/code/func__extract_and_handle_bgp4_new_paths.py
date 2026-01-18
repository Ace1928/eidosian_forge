from collections import namedtuple
import itertools
import logging
import socket
import time
import traceback
from os_ken.services.protocols.bgp.base import Activity
from os_ken.services.protocols.bgp.base import Sink
from os_ken.services.protocols.bgp.base import Source
from os_ken.services.protocols.bgp.base import SUPPORTED_GLOBAL_RF
from os_ken.services.protocols.bgp import constants as const
from os_ken.services.protocols.bgp.model import OutgoingRoute
from os_ken.services.protocols.bgp.model import SentRoute
from os_ken.services.protocols.bgp.info_base.base import PrefixFilter
from os_ken.services.protocols.bgp.info_base.base import AttributeMap
from os_ken.services.protocols.bgp.model import ReceivedRoute
from os_ken.services.protocols.bgp.net_ctrl import NET_CONTROLLER
from os_ken.services.protocols.bgp.rtconf.neighbors import NeighborConfListener
from os_ken.services.protocols.bgp.rtconf.neighbors import CONNECT_MODE_PASSIVE
from os_ken.services.protocols.bgp.signals.emit import BgpSignalBus
from os_ken.services.protocols.bgp.speaker import BgpProtocol
from os_ken.services.protocols.bgp.info_base.ipv4 import Ipv4Path
from os_ken.services.protocols.bgp.info_base.vpnv4 import Vpnv4Path
from os_ken.services.protocols.bgp.info_base.vpnv6 import Vpnv6Path
from os_ken.services.protocols.bgp.rtconf.vrfs import VRF_RF_IPV4, VRF_RF_IPV6
from os_ken.services.protocols.bgp.utils import bgp as bgp_utils
from os_ken.services.protocols.bgp.utils.evtlet import EventletIOFactory
from os_ken.services.protocols.bgp.utils import stats
from os_ken.services.protocols.bgp.utils.validation import is_valid_old_asn
from os_ken.lib.packet import bgp
from os_ken.lib.packet.bgp import RouteFamily
from os_ken.lib.packet.bgp import RF_IPv4_UC
from os_ken.lib.packet.bgp import RF_IPv6_UC
from os_ken.lib.packet.bgp import RF_IPv4_VPN
from os_ken.lib.packet.bgp import RF_IPv6_VPN
from os_ken.lib.packet.bgp import RF_IPv4_FLOWSPEC
from os_ken.lib.packet.bgp import RF_VPNv4_FLOWSPEC
from os_ken.lib.packet.bgp import RF_RTC_UC
from os_ken.lib.packet.bgp import get_rf
from os_ken.lib.packet.bgp import BGPOpen
from os_ken.lib.packet.bgp import BGPUpdate
from os_ken.lib.packet.bgp import BGPRouteRefresh
from os_ken.lib.packet.bgp import BGP_ERROR_CEASE
from os_ken.lib.packet.bgp import BGP_ERROR_SUB_ADMINISTRATIVE_SHUTDOWN
from os_ken.lib.packet.bgp import BGP_ERROR_SUB_CONNECTION_COLLISION_RESOLUTION
from os_ken.lib.packet.bgp import BGP_MSG_UPDATE
from os_ken.lib.packet.bgp import BGP_MSG_KEEPALIVE
from os_ken.lib.packet.bgp import BGP_MSG_ROUTE_REFRESH
from os_ken.lib.packet.bgp import BGPPathAttributeNextHop
from os_ken.lib.packet.bgp import BGPPathAttributeAsPath
from os_ken.lib.packet.bgp import BGPPathAttributeAs4Path
from os_ken.lib.packet.bgp import BGPPathAttributeLocalPref
from os_ken.lib.packet.bgp import BGPPathAttributeExtendedCommunities
from os_ken.lib.packet.bgp import BGPPathAttributeOriginatorId
from os_ken.lib.packet.bgp import BGPPathAttributeClusterList
from os_ken.lib.packet.bgp import BGPPathAttributeMpReachNLRI
from os_ken.lib.packet.bgp import BGPPathAttributeMpUnreachNLRI
from os_ken.lib.packet.bgp import BGPPathAttributeCommunities
from os_ken.lib.packet.bgp import BGPPathAttributeMultiExitDisc
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_ORIGIN
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_AGGREGATOR
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_AS4_AGGREGATOR
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_AS_PATH
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_AS4_PATH
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_NEXT_HOP
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_MP_REACH_NLRI
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_MP_UNREACH_NLRI
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_MULTI_EXIT_DISC
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_COMMUNITIES
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_ORIGINATOR_ID
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_CLUSTER_LIST
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_EXTENDED_COMMUNITIES
from os_ken.lib.packet.bgp import BGP_ATTR_TYEP_PMSI_TUNNEL_ATTRIBUTE
from os_ken.lib.packet.bgp import BGPTwoOctetAsSpecificExtendedCommunity
from os_ken.lib.packet.bgp import BGPIPv4AddressSpecificExtendedCommunity
from os_ken.lib.packet import safi as subaddr_family
def _extract_and_handle_bgp4_new_paths(self, update_msg):
    """Extracts new paths advertised in the given update message's
         *MpReachNlri* attribute.

        Assumes MPBGP capability is enabled and message was validated.
        Parameters:
            - update_msg: (Update) is assumed to be checked for all bgp
            message errors.
            - valid_rts: (iterable) current valid/configured RTs.

        Extracted paths are added to appropriate *Destination* for further
        processing.
        """
    umsg_pattrs = update_msg.pathattr_map
    next_hop = update_msg.get_path_attr(BGP_ATTR_TYPE_NEXT_HOP).value
    msg_nlri_list = update_msg.nlri
    if not msg_nlri_list:
        LOG.debug('Update message did not have any new MP_REACH_NLRIs.')
        return
    for msg_nlri in msg_nlri_list:
        LOG.debug('NLRI: %s', msg_nlri)
        new_path = bgp_utils.create_path(self, msg_nlri, pattrs=umsg_pattrs, nexthop=next_hop)
        LOG.debug('Extracted paths from Update msg.: %s', new_path)
        block, blocked_cause = self._apply_in_filter(new_path)
        nlri_str = new_path.nlri.formatted_nlri_str
        received_route = ReceivedRoute(new_path, self, block)
        self._adj_rib_in[nlri_str] = received_route
        self._signal_bus.adj_rib_in_changed(self, received_route)
        if not block:
            tm = self._core_service.table_manager
            tm.learn_path(new_path)
        else:
            LOG.debug('prefix : %s is blocked by in-bound filter: %s', msg_nlri, blocked_cause)
    if msg_nlri_list:
        self.state.incr(PeerCounterNames.RECV_PREFIXES, incr_by=len(msg_nlri_list))
        if self._neigh_conf.exceeds_max_prefix_allowed(self.state.get_count(PeerCounterNames.RECV_PREFIXES)):
            LOG.error('Max. prefix allowed for this neighbor exceeded.')