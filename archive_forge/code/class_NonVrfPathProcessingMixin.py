import abc
from abc import ABCMeta
from abc import abstractmethod
from copy import copy
import logging
import functools
import netaddr
from os_ken.lib.packet.bgp import RF_IPv4_UC
from os_ken.lib.packet.bgp import RouteTargetMembershipNLRI
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_EXTENDED_COMMUNITIES
from os_ken.lib.packet.bgp import BGPPathAttributeLocalPref
from os_ken.lib.packet.bgp import BGP_ATTR_TYPE_AS_PATH
from os_ken.services.protocols.bgp.base import OrderedDict
from os_ken.services.protocols.bgp.constants import VPN_TABLE
from os_ken.services.protocols.bgp.constants import VRF_TABLE
from os_ken.services.protocols.bgp.model import OutgoingRoute
from os_ken.services.protocols.bgp.processor import BPR_ONLY_PATH
from os_ken.services.protocols.bgp.processor import BPR_UNKNOWN
class NonVrfPathProcessingMixin(object):
    """Mixin reacting to best-path selection algorithm on main table
    level. Intended to use with "Destination" subclasses.
    Applies to most of Destinations except for VrfDest
    because they are processed at VRF level, so different logic applies.
    """

    def __init__(self):
        self._core_service = None
        self._known_path_list = []

    def _best_path_lost(self):
        self._best_path = None
        if self._sent_routes:
            for sent_route in self._sent_routes.values():
                sent_path = sent_route.path
                withdraw_clone = sent_path.clone(for_withdrawal=True)
                outgoing_route = OutgoingRoute(withdraw_clone)
                sent_route.sent_peer.enque_outgoing_msg(outgoing_route)
                LOG.debug('Sending withdrawal to %s for %s', sent_route.sent_peer, outgoing_route)
            self._sent_routes = {}

    def _new_best_path(self, new_best_path):
        old_best_path = self._best_path
        self._best_path = new_best_path
        LOG.debug('New best path selected for destination %s', self)
        if old_best_path and old_best_path not in self._known_path_list and self._sent_routes:
            self._sent_routes = {}
        pm = self._core_service.peer_manager
        pm.comm_new_best_to_bgp_peers(new_best_path)
        if old_best_path and self._sent_routes:
            for sent_route in self._sent_routes.values():
                sent_path = sent_route.path
                withdraw_clone = sent_path.clone(for_withdrawal=True)
                outgoing_route = OutgoingRoute(withdraw_clone)
                sent_route.sent_peer.enque_outgoing_msg(outgoing_route)
                LOG.debug('Sending withdrawal to %s for %s', sent_route.sent_peer, outgoing_route)
                self._sent_routes = {}