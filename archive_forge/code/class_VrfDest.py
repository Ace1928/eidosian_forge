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
class VrfDest(Destination, metaclass=abc.ABCMeta):
    """Base class for VRF destination."""

    def __init__(self, table, nlri):
        super(VrfDest, self).__init__(table, nlri)
        self._route_dist = self._table.vrf_conf.route_dist

    @property
    def nlri_str(self):
        return self._nlri.prefix

    def _best_path_lost(self):
        old_best_path = self._best_path
        self._best_path = None
        if old_best_path is None:
            return
        if old_best_path.source is not None:
            old_best_path = old_best_path.clone(for_withdrawal=True)
            self._core_service.update_flexinet_peers(old_best_path, self._route_dist)
        else:
            gpath = old_best_path.clone_to_vpn(self._route_dist, for_withdrawal=True)
            tm = self._core_service.table_manager
            tm.learn_path(gpath)

    def _new_best_path(self, best_path):
        LOG.debug('New best path selected for destination %s', self)
        old_best_path = self._best_path
        assert best_path != old_best_path
        self._best_path = best_path
        if best_path.source is not None:

            def really_diff():
                old_labels = old_best_path.label_list
                new_labels = best_path.label_list
                return old_best_path.nexthop != best_path.nexthop or set(old_labels) != set(new_labels)
            if not old_best_path or (old_best_path and really_diff()):
                self._core_service.update_flexinet_peers(best_path, self._route_dist)
        else:
            gpath = best_path.clone_to_vpn(self._route_dist)
            tm = self._core_service.table_manager
            tm.learn_path(gpath)
            LOG.debug('VRF table %s has new best path: %s', self._route_dist, self.best_path)

    def _remove_withdrawals(self):
        """Removes withdrawn paths.

        Note:
        We may have disproportionate number of withdraws compared to know paths
        since not all paths get installed into the table due to bgp policy and
        we can receive withdraws for such paths and withdrawals may not be
        stopped by the same policies.
        """
        LOG.debug('Removing %s withdrawals', len(self._withdraw_list))
        if not self._withdraw_list:
            return
        if not self._known_path_list:
            LOG.debug('Found %s withdrawals for path(s) that did not get installed.', len(self._withdraw_list))
            del self._withdraw_list[:]
            return
        matches = []
        w_matches = []
        for withdraw in self._withdraw_list:
            match = None
            for path in self._known_path_list:
                if path.puid == withdraw.puid:
                    match = path
                    matches.append(path)
                    w_matches.append(withdraw)
                    break
            if not match:
                LOG.debug('No matching path for withdraw found, may be path was not installed into table: %s', withdraw)
        if len(matches) != len(self._withdraw_list):
            LOG.debug('Did not find match for some withdrawals. Number of matches(%s), number of withdrawals (%s)', len(matches), len(self._withdraw_list))
        for match in matches:
            self._known_path_list.remove(match)
        for w_match in w_matches:
            self._withdraw_list.remove(w_match)

    def _remove_old_paths(self):
        """Identifies which of known paths are old and removes them.

        Known paths will no longer have paths whose new version is present in
        new paths.
        """
        new_paths = self._new_path_list
        known_paths = self._known_path_list
        for new_path in new_paths:
            old_paths = []
            for path in known_paths:
                if new_path.puid == path.puid:
                    old_paths.append(path)
                    break
            for old_path in old_paths:
                known_paths.remove(old_path)
                LOG.debug('Implicit withdrawal of old path, since we have learned new path from same source: %s', old_path)

    def _validate_path(self, path):
        if not path or not hasattr(path, 'label_list'):
            raise ValueError('Invalid value of path. Expected type with attribute label_list got %s' % path)