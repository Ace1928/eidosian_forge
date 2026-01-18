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
class NeighborConf(ConfWithId, ConfWithStats):
    """Class that encapsulates one neighbors' configuration."""
    UPDATE_ENABLED_EVT = 'update_enabled_evt'
    UPDATE_MED_EVT = 'update_med_evt'
    UPDATE_CONNECT_MODE_EVT = 'update_connect_mode_evt'
    VALID_EVT = frozenset([UPDATE_ENABLED_EVT, UPDATE_MED_EVT, UPDATE_CONNECT_MODE_EVT])
    REQUIRED_SETTINGS = frozenset([REMOTE_AS, IP_ADDRESS])
    OPTIONAL_SETTINGS = frozenset([CAP_REFRESH, CAP_ENHANCED_REFRESH, CAP_FOUR_OCTET_AS_NUMBER, CAP_MBGP_IPV4, CAP_MBGP_IPV6, CAP_MBGP_VPNV4, CAP_MBGP_VPNV6, CAP_RTC, CAP_MBGP_EVPN, CAP_MBGP_IPV4FS, CAP_MBGP_VPNV4FS, CAP_MBGP_IPV6FS, CAP_MBGP_VPNV6FS, CAP_MBGP_L2VPNFS, RTC_AS, HOLD_TIME, REMOTE_PORT, ENABLED, MULTI_EXIT_DISC, MAX_PREFIXES, ADVERTISE_PEER_AS, SITE_OF_ORIGINS, LOCAL_ADDRESS, LOCAL_PORT, LOCAL_AS, PEER_NEXT_HOP, PASSWORD, IN_FILTER, OUT_FILTER, IS_ROUTE_SERVER_CLIENT, IS_ROUTE_REFLECTOR_CLIENT, CHECK_FIRST_AS, IS_NEXT_HOP_SELF, CONNECT_MODE])

    def __init__(self, **kwargs):
        super(NeighborConf, self).__init__(**kwargs)

    def _init_opt_settings(self, **kwargs):
        self._settings[CAP_REFRESH] = compute_optional_conf(CAP_REFRESH, DEFAULT_CAP_REFRESH, **kwargs)
        self._settings[CAP_ENHANCED_REFRESH] = compute_optional_conf(CAP_ENHANCED_REFRESH, DEFAULT_CAP_ENHANCED_REFRESH, **kwargs)
        self._settings[CAP_FOUR_OCTET_AS_NUMBER] = compute_optional_conf(CAP_FOUR_OCTET_AS_NUMBER, DEFAULT_CAP_FOUR_OCTET_AS_NUMBER, **kwargs)
        self._settings[CAP_MBGP_IPV4] = compute_optional_conf(CAP_MBGP_IPV4, DEFAULT_CAP_MBGP_IPV4, **kwargs)
        self._settings[CAP_MBGP_IPV6] = compute_optional_conf(CAP_MBGP_IPV6, DEFAULT_CAP_MBGP_IPV6, **kwargs)
        self._settings[CAP_MBGP_VPNV4] = compute_optional_conf(CAP_MBGP_VPNV4, DEFAULT_CAP_MBGP_VPNV4, **kwargs)
        self._settings[CAP_MBGP_EVPN] = compute_optional_conf(CAP_MBGP_EVPN, DEFAULT_CAP_MBGP_EVPN, **kwargs)
        self._settings[CAP_MBGP_VPNV6] = compute_optional_conf(CAP_MBGP_VPNV6, DEFAULT_CAP_MBGP_VPNV6, **kwargs)
        self._settings[CAP_MBGP_IPV4FS] = compute_optional_conf(CAP_MBGP_IPV4FS, DEFAULT_CAP_MBGP_IPV4FS, **kwargs)
        self._settings[CAP_MBGP_IPV6FS] = compute_optional_conf(CAP_MBGP_IPV6FS, DEFAULT_CAP_MBGP_IPV6FS, **kwargs)
        self._settings[CAP_MBGP_VPNV4FS] = compute_optional_conf(CAP_MBGP_VPNV4FS, DEFAULT_CAP_MBGP_VPNV4FS, **kwargs)
        self._settings[CAP_MBGP_VPNV6FS] = compute_optional_conf(CAP_MBGP_VPNV6FS, DEFAULT_CAP_MBGP_VPNV6FS, **kwargs)
        self._settings[CAP_MBGP_L2VPNFS] = compute_optional_conf(CAP_MBGP_L2VPNFS, DEFAULT_CAP_MBGP_L2VPNFS, **kwargs)
        self._settings[HOLD_TIME] = compute_optional_conf(HOLD_TIME, DEFAULT_HOLD_TIME, **kwargs)
        self._settings[ENABLED] = compute_optional_conf(ENABLED, DEFAULT_ENABLED, **kwargs)
        self._settings[MAX_PREFIXES] = compute_optional_conf(MAX_PREFIXES, DEFAULT_MAX_PREFIXES, **kwargs)
        self._settings[ADVERTISE_PEER_AS] = compute_optional_conf(ADVERTISE_PEER_AS, DEFAULT_ADVERTISE_PEER_AS, **kwargs)
        self._settings[IN_FILTER] = compute_optional_conf(IN_FILTER, DEFAULT_IN_FILTER, **kwargs)
        self._settings[OUT_FILTER] = compute_optional_conf(OUT_FILTER, DEFAULT_OUT_FILTER, **kwargs)
        self._settings[IS_ROUTE_SERVER_CLIENT] = compute_optional_conf(IS_ROUTE_SERVER_CLIENT, DEFAULT_IS_ROUTE_SERVER_CLIENT, **kwargs)
        self._settings[IS_ROUTE_REFLECTOR_CLIENT] = compute_optional_conf(IS_ROUTE_REFLECTOR_CLIENT, DEFAULT_IS_ROUTE_REFLECTOR_CLIENT, **kwargs)
        self._settings[CHECK_FIRST_AS] = compute_optional_conf(CHECK_FIRST_AS, DEFAULT_CHECK_FIRST_AS, **kwargs)
        self._settings[IS_NEXT_HOP_SELF] = compute_optional_conf(IS_NEXT_HOP_SELF, DEFAULT_IS_NEXT_HOP_SELF, **kwargs)
        self._settings[CONNECT_MODE] = compute_optional_conf(CONNECT_MODE, DEFAULT_CONNECT_MODE, **kwargs)
        self._settings[REMOTE_PORT] = compute_optional_conf(REMOTE_PORT, DEFAULT_BGP_PORT, **kwargs)
        med = kwargs.pop(MULTI_EXIT_DISC, None)
        if med and validate_med(med):
            self._settings[MULTI_EXIT_DISC] = med
        soos = kwargs.pop(SITE_OF_ORIGINS, None)
        if soos and validate_soo_list(soos):
            self._settings[SITE_OF_ORIGINS] = soos
        self._settings[LOCAL_ADDRESS] = compute_optional_conf(LOCAL_ADDRESS, None, **kwargs)
        self._settings[LOCAL_PORT] = compute_optional_conf(LOCAL_PORT, None, **kwargs)
        from os_ken.services.protocols.bgp.core_manager import CORE_MANAGER
        g_local_as = CORE_MANAGER.common_conf.local_as
        self._settings[LOCAL_AS] = compute_optional_conf(LOCAL_AS, g_local_as, **kwargs)
        self._settings[PEER_NEXT_HOP] = compute_optional_conf(PEER_NEXT_HOP, None, **kwargs)
        self._settings[PASSWORD] = compute_optional_conf(PASSWORD, None, **kwargs)
        self._settings[CAP_RTC] = compute_optional_conf(CAP_RTC, DEFAULT_CAP_RTC, **kwargs)
        self._settings[RTC_AS] = compute_optional_conf(RTC_AS, g_local_as, **kwargs)
        super(NeighborConf, self)._init_opt_settings(**kwargs)

    @classmethod
    def get_opt_settings(cls):
        self_confs = super(NeighborConf, cls).get_opt_settings()
        self_confs.update(NeighborConf.OPTIONAL_SETTINGS)
        return self_confs

    @classmethod
    def get_req_settings(cls):
        self_confs = super(NeighborConf, cls).get_req_settings()
        self_confs.update(NeighborConf.REQUIRED_SETTINGS)
        return self_confs

    @classmethod
    def get_valid_evts(cls):
        self_valid_evts = super(NeighborConf, cls).get_valid_evts()
        self_valid_evts.update(NeighborConf.VALID_EVT)
        return self_valid_evts

    @property
    def remote_as(self):
        return self._settings[REMOTE_AS]

    @property
    def ip_address(self):
        return self._settings[IP_ADDRESS]

    @property
    def port(self):
        return self._settings[REMOTE_PORT]

    @property
    def host_bind_ip(self):
        return self._settings[LOCAL_ADDRESS]

    @property
    def host_bind_port(self):
        return self._settings[LOCAL_PORT]

    @property
    def next_hop(self):
        return self._settings[PEER_NEXT_HOP]

    @property
    def password(self):
        return self._settings[PASSWORD]

    @property
    def local_as(self):
        return self._settings[LOCAL_AS]

    @property
    def hold_time(self):
        return self._settings[HOLD_TIME]

    @property
    def cap_refresh(self):
        return self._settings[CAP_REFRESH]

    @property
    def cap_enhanced_refresh(self):
        return self._settings[CAP_ENHANCED_REFRESH]

    @property
    def cap_four_octet_as_number(self):
        return self._settings[CAP_FOUR_OCTET_AS_NUMBER]

    @cap_four_octet_as_number.setter
    def cap_four_octet_as_number(self, cap):
        kwargs = {CAP_FOUR_OCTET_AS_NUMBER: cap}
        self._settings[CAP_FOUR_OCTET_AS_NUMBER] = compute_optional_conf(CAP_FOUR_OCTET_AS_NUMBER, DEFAULT_CAP_FOUR_OCTET_AS_NUMBER, **kwargs)

    @property
    def cap_mbgp_ipv4(self):
        return self._settings[CAP_MBGP_IPV4]

    @property
    def cap_mbgp_ipv6(self):
        return self._settings[CAP_MBGP_IPV6]

    @property
    def cap_mbgp_vpnv4(self):
        return self._settings[CAP_MBGP_VPNV4]

    @property
    def cap_mbgp_vpnv6(self):
        return self._settings[CAP_MBGP_VPNV6]

    @property
    def cap_mbgp_evpn(self):
        return self._settings[CAP_MBGP_EVPN]

    @property
    def cap_mbgp_ipv4fs(self):
        return self._settings[CAP_MBGP_IPV4FS]

    @property
    def cap_mbgp_ipv6fs(self):
        return self._settings[CAP_MBGP_IPV6FS]

    @property
    def cap_mbgp_vpnv4fs(self):
        return self._settings[CAP_MBGP_VPNV4FS]

    @property
    def cap_mbgp_vpnv6fs(self):
        return self._settings[CAP_MBGP_VPNV6FS]

    @property
    def cap_mbgp_l2vpnfs(self):
        return self._settings[CAP_MBGP_L2VPNFS]

    @property
    def cap_rtc(self):
        return self._settings[CAP_RTC]

    @property
    def enabled(self):
        return self._settings[ENABLED]

    @enabled.setter
    def enabled(self, enable):
        if self._settings[ENABLED] != enable:
            self._settings[ENABLED] = enable
            self._notify_listeners(NeighborConf.UPDATE_ENABLED_EVT, enable)

    @property
    def multi_exit_disc(self):
        return self._settings.get(MULTI_EXIT_DISC)

    @multi_exit_disc.setter
    def multi_exit_disc(self, value):
        if self._settings.get(MULTI_EXIT_DISC) != value:
            self._settings[MULTI_EXIT_DISC] = value
            self._notify_listeners(NeighborConf.UPDATE_MED_EVT, value)

    @property
    def soo_list(self):
        soos = self._settings.get(SITE_OF_ORIGINS)
        if soos:
            soos = list(soos)
        else:
            soos = []
        return soos

    @property
    def rtc_as(self):
        return self._settings[RTC_AS]

    @property
    def in_filter(self):
        return self._settings[IN_FILTER]

    @property
    def out_filter(self):
        return self._settings[OUT_FILTER]

    @property
    def is_route_server_client(self):
        return self._settings[IS_ROUTE_SERVER_CLIENT]

    @property
    def is_route_reflector_client(self):
        return self._settings[IS_ROUTE_REFLECTOR_CLIENT]

    @property
    def check_first_as(self):
        return self._settings[CHECK_FIRST_AS]

    @property
    def is_next_hop_self(self):
        return self._settings[IS_NEXT_HOP_SELF]

    @property
    def connect_mode(self):
        return self._settings[CONNECT_MODE]

    @connect_mode.setter
    def connect_mode(self, mode):
        self._settings[CONNECT_MODE] = mode
        self._notify_listeners(NeighborConf.UPDATE_CONNECT_MODE_EVT, mode)

    def exceeds_max_prefix_allowed(self, prefix_count):
        allowed_max = self._settings[MAX_PREFIXES]
        does_exceed = False
        if allowed_max != 0:
            if prefix_count > allowed_max:
                does_exceed = True
        return does_exceed

    def get_configured_capabilities(self):
        """Returns configured capabilities."""
        capabilities = OrderedDict()
        mbgp_caps = []
        if self.cap_mbgp_ipv4:
            mbgp_caps.append(BGPOptParamCapabilityMultiprotocol(RF_IPv4_UC.afi, RF_IPv4_UC.safi))
        if self.cap_mbgp_ipv6:
            mbgp_caps.append(BGPOptParamCapabilityMultiprotocol(RF_IPv6_UC.afi, RF_IPv6_UC.safi))
        if self.cap_mbgp_vpnv4:
            mbgp_caps.append(BGPOptParamCapabilityMultiprotocol(RF_IPv4_VPN.afi, RF_IPv4_VPN.safi))
        if self.cap_mbgp_vpnv6:
            mbgp_caps.append(BGPOptParamCapabilityMultiprotocol(RF_IPv6_VPN.afi, RF_IPv6_VPN.safi))
        if self.cap_rtc:
            mbgp_caps.append(BGPOptParamCapabilityMultiprotocol(RF_RTC_UC.afi, RF_RTC_UC.safi))
        if self.cap_mbgp_evpn:
            mbgp_caps.append(BGPOptParamCapabilityMultiprotocol(RF_L2_EVPN.afi, RF_L2_EVPN.safi))
        if self.cap_mbgp_ipv4fs:
            mbgp_caps.append(BGPOptParamCapabilityMultiprotocol(RF_IPv4_FLOWSPEC.afi, RF_IPv4_FLOWSPEC.safi))
        if self.cap_mbgp_ipv6fs:
            mbgp_caps.append(BGPOptParamCapabilityMultiprotocol(RF_IPv6_FLOWSPEC.afi, RF_IPv6_FLOWSPEC.safi))
        if self.cap_mbgp_vpnv4fs:
            mbgp_caps.append(BGPOptParamCapabilityMultiprotocol(RF_VPNv4_FLOWSPEC.afi, RF_VPNv4_FLOWSPEC.safi))
        if self.cap_mbgp_vpnv6fs:
            mbgp_caps.append(BGPOptParamCapabilityMultiprotocol(RF_VPNv6_FLOWSPEC.afi, RF_VPNv6_FLOWSPEC.safi))
        if self.cap_mbgp_l2vpnfs:
            mbgp_caps.append(BGPOptParamCapabilityMultiprotocol(RF_L2VPN_FLOWSPEC.afi, RF_L2VPN_FLOWSPEC.safi))
        if mbgp_caps:
            capabilities[BGP_CAP_MULTIPROTOCOL] = mbgp_caps
        if self.cap_refresh:
            capabilities[BGP_CAP_ROUTE_REFRESH] = [BGPOptParamCapabilityRouteRefresh()]
        if self.cap_enhanced_refresh:
            capabilities[BGP_CAP_ENHANCED_ROUTE_REFRESH] = [BGPOptParamCapabilityEnhancedRouteRefresh()]
        if self.cap_four_octet_as_number:
            capabilities[BGP_CAP_FOUR_OCTET_AS_NUMBER] = [BGPOptParamCapabilityFourOctetAsNumber(self.local_as)]
        return capabilities

    def __repr__(self):
        return '<%s(%r, %r, %r)>' % (self.__class__.__name__, self.remote_as, self.ip_address, self.enabled)

    def __str__(self):
        return 'Neighbor: %s' % self.ip_address