import netaddr
from os_ken.lib import hub
from os_ken.lib import ip
from os_ken.lib.packet.bgp import (
from os_ken.services.protocols.bgp.core_manager import CORE_MANAGER
from os_ken.services.protocols.bgp.signals.emit import BgpSignalBus
from os_ken.services.protocols.bgp.api.base import call
from os_ken.services.protocols.bgp.api.base import PREFIX
from os_ken.services.protocols.bgp.api.base import EVPN_ROUTE_TYPE
from os_ken.services.protocols.bgp.api.base import EVPN_ESI
from os_ken.services.protocols.bgp.api.base import EVPN_ETHERNET_TAG_ID
from os_ken.services.protocols.bgp.api.base import REDUNDANCY_MODE
from os_ken.services.protocols.bgp.api.base import IP_ADDR
from os_ken.services.protocols.bgp.api.base import MAC_ADDR
from os_ken.services.protocols.bgp.api.base import NEXT_HOP
from os_ken.services.protocols.bgp.api.base import IP_PREFIX
from os_ken.services.protocols.bgp.api.base import GW_IP_ADDR
from os_ken.services.protocols.bgp.api.base import ROUTE_DISTINGUISHER
from os_ken.services.protocols.bgp.api.base import ROUTE_FAMILY
from os_ken.services.protocols.bgp.api.base import EVPN_VNI
from os_ken.services.protocols.bgp.api.base import TUNNEL_TYPE
from os_ken.services.protocols.bgp.api.base import PMSI_TUNNEL_TYPE
from os_ken.services.protocols.bgp.api.base import MAC_MOBILITY
from os_ken.services.protocols.bgp.api.base import TUNNEL_ENDPOINT_IP
from os_ken.services.protocols.bgp.api.prefix import EVPN_MAX_ET
from os_ken.services.protocols.bgp.api.prefix import ESI_TYPE_LACP
from os_ken.services.protocols.bgp.api.prefix import ESI_TYPE_L2_BRIDGE
from os_ken.services.protocols.bgp.api.prefix import ESI_TYPE_MAC_BASED
from os_ken.services.protocols.bgp.api.prefix import EVPN_ETH_AUTO_DISCOVERY
from os_ken.services.protocols.bgp.api.prefix import EVPN_MAC_IP_ADV_ROUTE
from os_ken.services.protocols.bgp.api.prefix import EVPN_MULTICAST_ETAG_ROUTE
from os_ken.services.protocols.bgp.api.prefix import EVPN_ETH_SEGMENT
from os_ken.services.protocols.bgp.api.prefix import EVPN_IP_PREFIX_ROUTE
from os_ken.services.protocols.bgp.api.prefix import REDUNDANCY_MODE_ALL_ACTIVE
from os_ken.services.protocols.bgp.api.prefix import REDUNDANCY_MODE_SINGLE_ACTIVE
from os_ken.services.protocols.bgp.api.prefix import TUNNEL_TYPE_VXLAN
from os_ken.services.protocols.bgp.api.prefix import TUNNEL_TYPE_NVGRE
from os_ken.services.protocols.bgp.api.prefix import (
from os_ken.services.protocols.bgp.api.prefix import (
from os_ken.services.protocols.bgp.model import ReceivedRoute
from os_ken.services.protocols.bgp.rtconf.common import LOCAL_AS
from os_ken.services.protocols.bgp.rtconf.common import ROUTER_ID
from os_ken.services.protocols.bgp.rtconf.common import CLUSTER_ID
from os_ken.services.protocols.bgp.rtconf.common import BGP_SERVER_HOSTS
from os_ken.services.protocols.bgp.rtconf.common import BGP_SERVER_PORT
from os_ken.services.protocols.bgp.rtconf.common import DEFAULT_BGP_SERVER_HOSTS
from os_ken.services.protocols.bgp.rtconf.common import DEFAULT_BGP_SERVER_PORT
from os_ken.services.protocols.bgp.rtconf.common import (
from os_ken.services.protocols.bgp.rtconf.common import DEFAULT_LABEL_RANGE
from os_ken.services.protocols.bgp.rtconf.common import REFRESH_MAX_EOR_TIME
from os_ken.services.protocols.bgp.rtconf.common import REFRESH_STALEPATH_TIME
from os_ken.services.protocols.bgp.rtconf.common import LABEL_RANGE
from os_ken.services.protocols.bgp.rtconf.common import ALLOW_LOCAL_AS_IN_COUNT
from os_ken.services.protocols.bgp.rtconf.common import LOCAL_PREF
from os_ken.services.protocols.bgp.rtconf.common import DEFAULT_LOCAL_PREF
from os_ken.services.protocols.bgp.rtconf import neighbors
from os_ken.services.protocols.bgp.rtconf import vrfs
from os_ken.services.protocols.bgp.rtconf.base import CAP_MBGP_IPV4
from os_ken.services.protocols.bgp.rtconf.base import CAP_MBGP_IPV6
from os_ken.services.protocols.bgp.rtconf.base import CAP_MBGP_VPNV4
from os_ken.services.protocols.bgp.rtconf.base import CAP_MBGP_VPNV6
from os_ken.services.protocols.bgp.rtconf.base import CAP_MBGP_EVPN
from os_ken.services.protocols.bgp.rtconf.base import CAP_MBGP_IPV4FS
from os_ken.services.protocols.bgp.rtconf.base import CAP_MBGP_IPV6FS
from os_ken.services.protocols.bgp.rtconf.base import CAP_MBGP_VPNV4FS
from os_ken.services.protocols.bgp.rtconf.base import CAP_MBGP_VPNV6FS
from os_ken.services.protocols.bgp.rtconf.base import CAP_MBGP_L2VPNFS
from os_ken.services.protocols.bgp.rtconf.base import CAP_ENHANCED_REFRESH
from os_ken.services.protocols.bgp.rtconf.base import CAP_FOUR_OCTET_AS_NUMBER
from os_ken.services.protocols.bgp.rtconf.base import MULTI_EXIT_DISC
from os_ken.services.protocols.bgp.rtconf.base import SITE_OF_ORIGINS
from os_ken.services.protocols.bgp.rtconf.neighbors import (
from os_ken.services.protocols.bgp.rtconf.vrfs import SUPPORTED_VRF_RF
from os_ken.services.protocols.bgp.info_base.base import Filter
from os_ken.services.protocols.bgp.info_base.ipv4 import Ipv4Path
from os_ken.services.protocols.bgp.info_base.ipv6 import Ipv6Path
from os_ken.services.protocols.bgp.info_base.vpnv4 import Vpnv4Path
from os_ken.services.protocols.bgp.info_base.vpnv6 import Vpnv6Path
from os_ken.services.protocols.bgp.info_base.evpn import EvpnPath
class EventPrefix(object):
    """
    Used to pass an update on any best remote path to
    best_path_change_handler.

    ================ ======================================================
    Attribute        Description
    ================ ======================================================
    remote_as        The AS number of a peer that caused this change
    route_dist       None in the case of IPv4 or IPv6 family
    prefix           A prefix was changed
    nexthop          The nexthop of the changed prefix
    label            MPLS label for VPNv4, VPNv6 or EVPN prefix
    path             An instance of ``info_base.base.Path`` subclass
    is_withdraw      True if this prefix has gone otherwise False
    ================ ======================================================
    """

    def __init__(self, path, is_withdraw):
        self.path = path
        self.is_withdraw = is_withdraw

    @property
    def remote_as(self):
        return self.path.source.remote_as

    @property
    def route_dist(self):
        if isinstance(self.path, Vpnv4Path) or isinstance(self.path, Vpnv6Path) or isinstance(self.path, EvpnPath):
            return self.path.nlri.route_dist
        else:
            return None

    @property
    def prefix(self):
        if isinstance(self.path, Ipv4Path) or isinstance(self.path, Ipv6Path):
            return self.path.nlri.addr + '/' + str(self.path.nlri.length)
        elif isinstance(self.path, Vpnv4Path) or isinstance(self.path, Vpnv6Path) or isinstance(self.path, EvpnPath):
            return self.path.nlri.prefix
        else:
            return None

    @property
    def nexthop(self):
        return self.path.nexthop

    @property
    def label(self):
        if isinstance(self.path, Vpnv4Path) or isinstance(self.path, Vpnv6Path) or isinstance(self.path, EvpnPath):
            return getattr(self.path.nlri, 'label_list', None)
        else:
            return None