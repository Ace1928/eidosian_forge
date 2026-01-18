import logging
import os
from os_ken import cfg
from os_ken.lib import hub
from os_ken.utils import load_source
from os_ken.base.app_manager import OSKenApp
from os_ken.controller.event import EventBase
from os_ken.services.protocols.bgp.base import add_bgp_error_metadata
from os_ken.services.protocols.bgp.base import BGPSException
from os_ken.services.protocols.bgp.base import BIN_ERROR
from os_ken.services.protocols.bgp.bgpspeaker import BGPSpeaker
from os_ken.services.protocols.bgp.net_ctrl import NET_CONTROLLER
from os_ken.services.protocols.bgp.net_ctrl import NC_RPC_BIND_IP
from os_ken.services.protocols.bgp.net_ctrl import NC_RPC_BIND_PORT
from os_ken.services.protocols.bgp.rtconf.base import RuntimeConfigError
from os_ken.services.protocols.bgp.rtconf.common import LOCAL_AS
from os_ken.services.protocols.bgp.rtconf.common import ROUTER_ID
from os_ken.services.protocols.bgp.utils.validation import is_valid_ipv4
from os_ken.services.protocols.bgp.utils.validation import is_valid_ipv6
class OSKenBGPSpeaker(OSKenApp):
    """
    Base application for implementing BGP applications.
    """
    _EVENTS = [EventBestPathChanged, EventAdjRibInChanged, EventPeerDown, EventPeerUp]

    def __init__(self, *args, **kwargs):
        super(OSKenBGPSpeaker, self).__init__(*args, **kwargs)
        self.config_file = CONF.config_file
        self.speaker = None

    def start(self):
        super(OSKenBGPSpeaker, self).start()
        if self.config_file:
            LOG.debug('Loading config file %s...', self.config_file)
            settings = load_config(self.config_file)
            if hasattr(settings, 'LOGGING'):
                LOG.debug('Loading LOGGING settings... (NOT implemented yet)')
            if hasattr(settings, 'BGP'):
                LOG.debug('Loading BGP settings...')
                self._start_speaker(settings.BGP)
            if hasattr(settings, 'SSH'):
                LOG.debug('Loading SSH settings...')
                from os_ken.services.protocols.bgp.operator import ssh
                hub.spawn(ssh.SSH_CLI_CONTROLLER.start, **settings.SSH)
        rpc_settings = {NC_RPC_BIND_PORT: CONF.rpc_port, NC_RPC_BIND_IP: validate_rpc_host(CONF.rpc_host)}
        return hub.spawn(NET_CONTROLLER.start, **rpc_settings)

    def _start_speaker(self, settings):
        """
        Starts BGPSpeaker using the given settings.
        """
        _required_settings = (LOCAL_AS, ROUTER_ID)
        for required in _required_settings:
            if required not in settings:
                raise ApplicationException(desc='Required BGP configuration missing: %s' % required)
        settings.setdefault('best_path_change_handler', self._notify_best_path_changed_event)
        settings.setdefault('adj_rib_in_change_handler', self._notify_adj_rib_in_changed_event)
        settings.setdefault('peer_down_handler', self._notify_peer_down_event)
        settings.setdefault('peer_up_handler', self._notify_peer_up_event)
        neighbors_settings = settings.pop('neighbors', [])
        vrfs_settings = settings.pop('vrfs', [])
        routes_settings = settings.pop('routes', [])
        LOG.debug('Starting BGPSpeaker...')
        settings.setdefault('as_number', settings.pop(LOCAL_AS))
        self.speaker = BGPSpeaker(**settings)
        LOG.debug('Adding neighbors...')
        self._add_neighbors(neighbors_settings)
        LOG.debug('Adding VRFs...')
        self._add_vrfs(vrfs_settings)
        LOG.debug('Adding routes...')
        self._add_routes(routes_settings)

    def _notify_best_path_changed_event(self, ev):
        ev = EventBestPathChanged(ev.path, ev.is_withdraw)
        self.send_event_to_observers(ev)

    def _notify_adj_rib_in_changed_event(self, ev, peer_ip, peer_as):
        ev = EventAdjRibInChanged(ev.path, ev.is_withdraw, peer_ip, peer_as)
        self.send_event_to_observers(ev)

    def _notify_peer_down_event(self, remote_ip, remote_as):
        ev = EventPeerDown(remote_ip, remote_as)
        self.send_event_to_observers(ev)

    def _notify_peer_up_event(self, remote_ip, remote_as):
        ev = EventPeerUp(remote_ip, remote_as)
        self.send_event_to_observers(ev)

    def _add_neighbors(self, settings):
        """
        Add BGP neighbors from the given settings.

        All valid neighbors are loaded.
        Miss-configured neighbors are ignored and errors are logged.
        """
        for neighbor_settings in settings:
            LOG.debug('Adding neighbor settings: %s', neighbor_settings)
            try:
                self.speaker.neighbor_add(**neighbor_settings)
            except RuntimeConfigError as e:
                LOG.exception(e)

    def _add_vrfs(self, settings):
        """
        Add BGP VRFs from the given settings.

        All valid VRFs are loaded.
        Miss-configured VRFs are ignored and errors are logged.
        """
        for vrf_settings in settings:
            LOG.debug('Adding VRF settings: %s', vrf_settings)
            try:
                self.speaker.vrf_add(**vrf_settings)
            except RuntimeConfigError as e:
                LOG.exception(e)

    def _add_routes(self, settings):
        """
        Add BGP routes from given settings.

        All valid routes are loaded.
        Miss-configured routes are ignored and errors are logged.
        """
        for route_settings in settings:
            if 'prefix' in route_settings:
                prefix_add = self.speaker.prefix_add
            elif 'route_type' in route_settings:
                prefix_add = self.speaker.evpn_prefix_add
            elif 'flowspec_family' in route_settings:
                prefix_add = self.speaker.flowspec_prefix_add
            else:
                LOG.debug('Skip invalid route settings: %s', route_settings)
                continue
            LOG.debug('Adding route settings: %s', route_settings)
            try:
                prefix_add(**route_settings)
            except RuntimeConfigError as e:
                LOG.exception(e)