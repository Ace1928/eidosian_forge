from abc import abstractmethod
import logging
import numbers
import netaddr
from os_ken.lib import ip
from os_ken.lib.packet.bgp import RF_IPv4_UC
from os_ken.lib.packet.bgp import RF_IPv6_UC
from os_ken.lib.packet.bgp import RF_IPv4_VPN
from os_ken.lib.packet.bgp import RF_IPv6_VPN
from os_ken.lib.packet.bgp import RF_L2_EVPN
from os_ken.lib.packet.bgp import RF_IPv4_FLOWSPEC
from os_ken.lib.packet.bgp import RF_IPv6_FLOWSPEC
from os_ken.lib.packet.bgp import RF_VPNv4_FLOWSPEC
from os_ken.lib.packet.bgp import RF_VPNv6_FLOWSPEC
from os_ken.lib.packet.bgp import RF_L2VPN_FLOWSPEC
from os_ken.lib.packet.bgp import RF_RTC_UC
from os_ken.lib.packet.bgp import BGPOptParamCapabilityFourOctetAsNumber
from os_ken.lib.packet.bgp import BGPOptParamCapabilityEnhancedRouteRefresh
from os_ken.lib.packet.bgp import BGPOptParamCapabilityMultiprotocol
from os_ken.lib.packet.bgp import BGPOptParamCapabilityRouteRefresh
from os_ken.lib.packet.bgp import BGP_CAP_FOUR_OCTET_AS_NUMBER
from os_ken.lib.packet.bgp import BGP_CAP_ENHANCED_ROUTE_REFRESH
from os_ken.lib.packet.bgp import BGP_CAP_MULTIPROTOCOL
from os_ken.lib.packet.bgp import BGP_CAP_ROUTE_REFRESH
from os_ken.services.protocols.bgp.base import OrderedDict
from os_ken.services.protocols.bgp.constants import STD_BGP_SERVER_PORT_NUM
from os_ken.services.protocols.bgp.rtconf.base import ADVERTISE_PEER_AS
from os_ken.services.protocols.bgp.rtconf.base import BaseConf
from os_ken.services.protocols.bgp.rtconf.base import BaseConfListener
from os_ken.services.protocols.bgp.rtconf.base import CAP_ENHANCED_REFRESH
from os_ken.services.protocols.bgp.rtconf.base import CAP_FOUR_OCTET_AS_NUMBER
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
from os_ken.services.protocols.bgp.rtconf.base import CAP_REFRESH
from os_ken.services.protocols.bgp.rtconf.base import CAP_RTC
from os_ken.services.protocols.bgp.rtconf.base import compute_optional_conf
from os_ken.services.protocols.bgp.rtconf.base import ConfigTypeError
from os_ken.services.protocols.bgp.rtconf.base import ConfigValueError
from os_ken.services.protocols.bgp.rtconf.base import ConfWithId
from os_ken.services.protocols.bgp.rtconf.base import ConfWithIdListener
from os_ken.services.protocols.bgp.rtconf.base import ConfWithStats
from os_ken.services.protocols.bgp.rtconf.base import ConfWithStatsListener
from os_ken.services.protocols.bgp.rtconf.base import HOLD_TIME
from os_ken.services.protocols.bgp.rtconf.base import MAX_PREFIXES
from os_ken.services.protocols.bgp.rtconf.base import MULTI_EXIT_DISC
from os_ken.services.protocols.bgp.rtconf.base import RTC_AS
from os_ken.services.protocols.bgp.rtconf.base import RuntimeConfigError
from os_ken.services.protocols.bgp.rtconf.base import SITE_OF_ORIGINS
from os_ken.services.protocols.bgp.rtconf.base import validate
from os_ken.services.protocols.bgp.rtconf.base import validate_med
from os_ken.services.protocols.bgp.rtconf.base import validate_soo_list
from os_ken.services.protocols.bgp.utils.validation import is_valid_asn
from os_ken.services.protocols.bgp.info_base.base import Filter
from os_ken.services.protocols.bgp.info_base.base import PrefixFilter
from os_ken.services.protocols.bgp.info_base.base import AttributeMap
class NeighborsConf(BaseConf):
    """Container of all neighbor configurations."""
    ADD_NEIGH_CONF_EVT = 'add_neigh_conf_evt'
    REMOVE_NEIGH_CONF_EVT = 'remove_neigh_conf_evt'
    VALID_EVT = frozenset([ADD_NEIGH_CONF_EVT, REMOVE_NEIGH_CONF_EVT])

    def __init__(self):
        super(NeighborsConf, self).__init__()
        self._neighbors = {}

    def _init_opt_settings(self, **kwargs):
        pass

    def update(self, **kwargs):
        raise NotImplementedError('Use either add/remove_neighbor_conf methods instead.')

    @property
    def rtc_as_set(self):
        """Returns current RTC AS configured for current neighbors.
        """
        rtc_as_set = set()
        for neigh in self._neighbors.values():
            rtc_as_set.add(neigh.rtc_as)
        return rtc_as_set

    @classmethod
    def get_valid_evts(cls):
        self_valid_evts = super(NeighborsConf, cls).get_valid_evts()
        self_valid_evts.update(NeighborsConf.VALID_EVT)
        return self_valid_evts

    def add_neighbor_conf(self, neigh_conf):
        if neigh_conf.ip_address in self._neighbors.keys():
            message = 'Neighbor with given ip address already exists'
            raise RuntimeConfigError(desc=message)
        self._neighbors[neigh_conf.ip_address] = neigh_conf
        self._notify_listeners(NeighborsConf.ADD_NEIGH_CONF_EVT, neigh_conf)

    def remove_neighbor_conf(self, neigh_ip_address):
        neigh_conf = self._neighbors.pop(neigh_ip_address, None)
        if not neigh_conf:
            raise RuntimeConfigError(desc='Tried to remove a neighbor that does not exists')
        else:
            self._notify_listeners(NeighborsConf.REMOVE_NEIGH_CONF_EVT, neigh_conf)
        return neigh_conf

    def get_neighbor_conf(self, neigh_ip_address):
        return self._neighbors.get(neigh_ip_address, None)

    def __repr__(self):
        return '<%s(%r)>' % (self.__class__.__name__, self._neighbors)

    def __str__(self):
        return "'Neighbors': %s" % self._neighbors

    @property
    def settings(self):
        return [neighbor.settings for _, neighbor in self._neighbors.items()]