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
def _extract_and_reconstruct_as_path(self, update_msg):
    """Extracts advertised AS path attributes in the given update message
        and reconstructs AS_PATH from AS_PATH and AS4_PATH if needed."""
    umsg_pattrs = update_msg.pathattr_map
    as_aggregator = umsg_pattrs.get(BGP_ATTR_TYPE_AGGREGATOR, None)
    as4_aggregator = umsg_pattrs.get(BGP_ATTR_TYPE_AS4_AGGREGATOR, None)
    if as_aggregator and as4_aggregator:
        if as_aggregator.as_number != bgp.AS_TRANS:
            update_msg.path_attributes.remove(as4_aggregator)
            as4_path = umsg_pattrs.pop(BGP_ATTR_TYPE_AS4_PATH, None)
            if as4_path:
                update_msg.path_attributes.remove(as4_path)
        else:
            update_msg.path_attributes.remove(as_aggregator)
            update_msg.path_attributes.remove(as4_aggregator)
            update_msg.path_attributes.append(bgp.BGPPathAttributeAggregator(as_number=as4_aggregator.as_number, addr=as4_aggregator.addr))
    as_path = umsg_pattrs.get(BGP_ATTR_TYPE_AS_PATH, None)
    as4_path = umsg_pattrs.get(BGP_ATTR_TYPE_AS4_PATH, None)
    if as_path and as4_path:
        if as_path.get_as_path_len() < as4_path.get_as_path_len():
            update_msg.path_attributes.remove(as4_path)
        else:
            update_msg.path_attributes.remove(as_path)
            update_msg.path_attributes.remove(as4_path)
            as_path = self._construct_as_path_attr(as_path, as4_path)
            update_msg.path_attributes.append(as_path)