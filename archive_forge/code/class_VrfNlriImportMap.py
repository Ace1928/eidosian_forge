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
class VrfNlriImportMap(ImportMap):
    VRF_PATH_CLASS = None
    NLRI_CLASS = None

    def __init__(self, prefix):
        assert self.VRF_PATH_CLASS is not None
        assert self.NLRI_CLASS is not None
        self._nlri = self.NLRI_CLASS(prefix)

    def match(self, vrf_path):
        if vrf_path.route_family != self.VRF_PATH_CLASS.ROUTE_FAMILY:
            LOG.error('vrf_paths route_family does not match importmapsroute_family. Applied to wrong table?')
            return False
        return vrf_path.nlri == self._nlri