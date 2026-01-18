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
class PeerState(object):
    """A BGP neighbor state. Think of this class as of information and stats
    container for Peer.
    """

    def __init__(self, peer, signal_bus):
        self.peer = peer
        self._bgp_state = const.BGP_FSM_IDLE
        self._established_time = 0
        self._last_bgp_error = None
        self.counters = {'recv_prefixes': 0, 'recv_updates': 0, 'sent_updates': 0, 'recv_notification': 0, 'sent_notification': 0, 'sent_refresh': 0, 'recv_refresh': 0, 'fms_established_transitions': 0}
        self._signal_bus = signal_bus
        self._signal_bus.register_listener(('error', 'bgp', self.peer), self._remember_last_bgp_error)
        self._signal_bus.register_listener(BgpSignalBus.BGP_NOTIFICATION_RECEIVED + (self.peer,), lambda _, msg: self.incr(PeerCounterNames.RECV_NOTIFICATION))
        self._signal_bus.register_listener(BgpSignalBus.BGP_NOTIFICATION_SENT + (self.peer,), lambda _, msg: self.incr(PeerCounterNames.SENT_NOTIFICATION))

    def _remember_last_bgp_error(self, identifier, data):
        self._last_bgp_error = dict([(k, v) for k, v in data.items() if k != 'peer'])

    @property
    def recv_prefix(self):
        return self.counters[PeerCounterNames.RECV_PREFIXES]

    @property
    def bgp_state(self):
        return self._bgp_state

    @bgp_state.setter
    def bgp_state(self, new_state):
        old_state = self._bgp_state
        if old_state == new_state:
            return
        self._bgp_state = new_state
        NET_CONTROLLER.send_rpc_notification('neighbor.state', {'ip_address': self.peer.ip_address, 'state': new_state})
        if new_state == const.BGP_FSM_ESTABLISHED:
            self.incr(PeerCounterNames.FSM_ESTB_TRANSITIONS)
            self._established_time = time.time()
            self._signal_bus.adj_up(self.peer)
            NET_CONTROLLER.send_rpc_notification('neighbor.up', {'ip_address': self.peer.ip_address})
        elif old_state == const.BGP_FSM_ESTABLISHED:
            self._established_time = 0
            self._signal_bus.adj_down(self.peer)
            NET_CONTROLLER.send_rpc_notification('neighbor.down', {'ip_address': self.peer.ip_address})
        LOG.debug('Peer %s BGP FSM went from %s to %s', self.peer.ip_address, old_state, self.bgp_state)

    def incr(self, counter_name, incr_by=1):
        if counter_name not in self.counters:
            raise ValueError('Un-recognized counter name: %s' % counter_name)
        counter = self.counters.setdefault(counter_name, 0)
        counter += incr_by
        self.counters[counter_name] = counter

    def get_count(self, counter_name):
        if counter_name not in self.counters:
            raise ValueError('Un-recognized counter name: %s' % counter_name)
        return self.counters.get(counter_name, 0)

    @property
    def total_msg_sent(self):
        """Returns total number of UPDATE, NOTIFICATION and ROUTE_REFRESH
         message sent to this peer.
         """
        return self.get_count(PeerCounterNames.SENT_REFRESH) + self.get_count(PeerCounterNames.SENT_UPDATES)

    @property
    def total_msg_recv(self):
        """Returns total number of UPDATE, NOTIFICATION and ROUTE_REFRESH
        messages received from this peer.
        """
        return self.get_count(PeerCounterNames.RECV_UPDATES) + self.get_count(PeerCounterNames.RECV_REFRESH) + self.get_count(PeerCounterNames.RECV_NOTIFICATION)

    def get_stats_summary_dict(self):
        """Returns basic stats.

        Returns a `dict` with various counts and stats, see below.
        """
        uptime = time.time() - self._established_time if self._established_time != 0 else -1
        return {stats.UPDATE_MSG_IN: self.get_count(PeerCounterNames.RECV_UPDATES), stats.UPDATE_MSG_OUT: self.get_count(PeerCounterNames.SENT_UPDATES), stats.TOTAL_MSG_IN: self.total_msg_recv, stats.TOTAL_MSG_OUT: self.total_msg_sent, stats.FMS_EST_TRANS: self.get_count(PeerCounterNames.FSM_ESTB_TRANSITIONS), stats.UPTIME: uptime}