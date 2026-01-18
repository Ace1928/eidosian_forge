import abc
import logging
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_ORIGIN
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_AS_PATH
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_EXTENDED_COMMUNITIES
from os_ken.lib.packet.bgp import BGP_ATTR_TYEP_PMSI_TUNNEL_ATTRIBUTE
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_MULTI_EXIT_DISC
from os_ken.lib.packet.bgp import BGPPathAttributeOrigin
from os_ken.lib.packet.bgp import BGPPathAttributeAsPath
from os_ken.lib.packet.bgp import EvpnEthernetSegmentNLRI
from os_ken.lib.packet.bgp import BGPPathAttributeExtendedCommunities
from os_ken.lib.packet.bgp import BGPPathAttributeMultiExitDisc
from os_ken.lib.packet.bgp import BGPEncapsulationExtendedCommunity
from os_ken.lib.packet.bgp import BGPEvpnEsiLabelExtendedCommunity
from os_ken.lib.packet.bgp import BGPEvpnEsImportRTExtendedCommunity
from os_ken.lib.packet.bgp import BGPPathAttributePmsiTunnel
from os_ken.lib.packet.bgp import PmsiTunnelIdIngressReplication
from os_ken.lib.packet.bgp import RF_L2_EVPN
from os_ken.lib.packet.bgp import EvpnMacIPAdvertisementNLRI
from os_ken.lib.packet.bgp import EvpnIpPrefixNLRI
from os_ken.lib.packet.safi import (
from os_ken.services.protocols.bgp.base import OrderedDict
from os_ken.services.protocols.bgp.constants import VPN_TABLE
from os_ken.services.protocols.bgp.constants import VRF_TABLE
from os_ken.services.protocols.bgp.info_base.base import Destination
from os_ken.services.protocols.bgp.info_base.base import Path
from os_ken.services.protocols.bgp.info_base.base import Table
from os_ken.services.protocols.bgp.utils.bgp import create_rt_extended_community
from os_ken.services.protocols.bgp.utils.stats import LOCAL_ROUTES
from os_ken.services.protocols.bgp.utils.stats import REMOTE_ROUTES
from os_ken.services.protocols.bgp.utils.stats import RESOURCE_ID
from os_ken.services.protocols.bgp.utils.stats import RESOURCE_NAME
class VrfPath(Path, metaclass=abc.ABCMeta):
    """Represents a way of reaching an IP destination with a VPN.
    """
    __slots__ = ('_label_list', '_puid')
    ROUTE_FAMILY = None
    VPN_PATH_CLASS = None
    VPN_NLRI_CLASS = None

    def __init__(self, puid, source, nlri, src_ver_num, pattrs=None, nexthop=None, is_withdraw=False, label_list=None):
        """Initializes a Vrf path.

            Parameters:
                - `puid`: (str) path ID, identifies VPN path from which this
                VRF path was imported.
                - `label_list`: (list) List of labels for this path.
            Note: other parameters are as documented in super class.
        """
        if self.ROUTE_FAMILY.safi in [IP_FLOWSPEC, VPN_FLOWSPEC]:
            nexthop = '0.0.0.0'
        Path.__init__(self, source, nlri, src_ver_num, pattrs, nexthop, is_withdraw)
        if label_list is None:
            label_list = []
        self._label_list = label_list
        self._puid = puid

    @property
    def puid(self):
        return self._puid

    @property
    def origin_rd(self):
        tokens = self.puid.split(':')
        return tokens[0] + ':' + tokens[1]

    @property
    def label_list(self):
        return self._label_list[:]

    @property
    def nlri_str(self):
        return self._nlri.prefix

    @staticmethod
    def create_puid(route_dist, ip_prefix):
        assert route_dist and ip_prefix
        return str(route_dist) + ':' + ip_prefix

    def clone(self, for_withdrawal=False):
        pathattrs = None
        if not for_withdrawal:
            pathattrs = self.pathattr_map
        clone = self.__class__(self.puid, self._source, self.nlri, self.source_version_num, pattrs=pathattrs, nexthop=self.nexthop, is_withdraw=for_withdrawal, label_list=self.label_list)
        return clone

    def clone_to_vpn(self, route_dist, for_withdrawal=False):
        if self.ROUTE_FAMILY == RF_L2_EVPN:
            vpn_nlri = self._nlri
        elif self.ROUTE_FAMILY.safi in [IP_FLOWSPEC, VPN_FLOWSPEC]:
            vpn_nlri = self.VPN_NLRI_CLASS(route_dist=route_dist, rules=self.nlri.rules)
        else:
            ip, masklen = self._nlri.prefix.split('/')
            vpn_nlri = self.VPN_NLRI_CLASS(length=int(masklen), addr=ip, labels=self.label_list, route_dist=route_dist)
        pathattrs = None
        if not for_withdrawal:
            pathattrs = self.pathattr_map
        vpnv_path = self.VPN_PATH_CLASS(source=self.source, nlri=vpn_nlri, src_ver_num=self.source_version_num, pattrs=pathattrs, nexthop=self.nexthop, is_withdraw=for_withdrawal)
        return vpnv_path

    def __eq__(self, b_path):
        if not isinstance(b_path, self.__class__):
            return False
        if not self.route_family == b_path.route_family:
            return False
        if not self.puid == b_path.puid:
            return False
        if not self.label_list == b_path.label_list:
            return False
        if not self.nexthop == b_path.nexthop:
            return False
        if not self.pathattr_map == b_path.pathattr_map:
            return False
        return True