import logging
import netaddr
import socket
from os_ken.lib.packet.bgp import BGP_ERROR_CEASE
from os_ken.lib.packet.bgp import BGP_ERROR_SUB_CONNECTION_RESET
from os_ken.lib.packet.bgp import BGP_ERROR_SUB_CONNECTION_COLLISION_RESOLUTION
from os_ken.lib.packet.bgp import RF_RTC_UC
from os_ken.lib.packet.bgp import BGP_ATTR_ORIGIN_INCOMPLETE
from os_ken.services.protocols.bgp.base import Activity
from os_ken.services.protocols.bgp.base import add_bgp_error_metadata
from os_ken.services.protocols.bgp.base import BGPSException
from os_ken.services.protocols.bgp.base import CORE_ERROR_CODE
from os_ken.services.protocols.bgp.constants import STD_BGP_SERVER_PORT_NUM
from os_ken.services.protocols.bgp import core_managers
from os_ken.services.protocols.bgp.model import FlexinetOutgoingRoute
from os_ken.services.protocols.bgp.protocol import Factory
from os_ken.services.protocols.bgp.signals.emit import BgpSignalBus
from os_ken.services.protocols.bgp.speaker import BgpProtocol
from os_ken.services.protocols.bgp.utils.rtfilter import RouteTargetManager
from os_ken.services.protocols.bgp.rtconf.neighbors import CONNECT_MODE_ACTIVE
from os_ken.services.protocols.bgp.utils import stats
from os_ken.services.protocols.bgp.bmp import BMPClient
from os_ken.lib import sockopt
from os_ken.lib import ip
class CoreService(Factory, Activity):
    """A service that maintains eBGP/iBGP sessions with BGP peers.

    Two instances of this class don't share any BGP state with each
    other. Manages peers, tables for various address-families, etc.
    """
    protocol = BgpProtocol

    def __init__(self, common_conf, neighbors_conf, vrfs_conf):
        self._common_config = common_conf
        self._neighbors_conf = neighbors_conf
        self._vrfs_conf = vrfs_conf
        Activity.__init__(self, name='core_service')
        self._signal_bus = BgpSignalBus()
        self._init_signal_listeners()
        self._rt_mgr = RouteTargetManager(self, neighbors_conf, vrfs_conf)
        self._table_manager = core_managers.TableCoreManager(self, common_conf)
        self._importmap_manager = core_managers.ImportMapManager()
        self._asn = self._common_config.local_as
        self._peer_manager = core_managers.PeerManager(self, self._neighbors_conf)
        self._sinks = set()
        self._conf_manager = core_managers.ConfigurationManager(self, common_conf, vrfs_conf, neighbors_conf)
        from os_ken.services.protocols.bgp.net_ctrl import NET_CONTROLLER
        self.register_flexinet_sink(NET_CONTROLLER)
        self.rf_state = {}
        self.client_factory = None
        self.server_factory = None
        self._next_hop_label = {}
        self._bgp_processor = None
        self.bmpclients = {}

    def _init_signal_listeners(self):
        self._signal_bus.register_listener(BgpSignalBus.BGP_DEST_CHANGED, lambda _, dest: self.enqueue_for_bgp_processing(dest))
        self._signal_bus.register_listener(BgpSignalBus.BGP_VRF_REMOVED, lambda _, route_dist: self.on_vrf_removed(route_dist))
        self._signal_bus.register_listener(BgpSignalBus.BGP_VRF_ADDED, lambda _, vrf_conf: self.on_vrf_added(vrf_conf))
        self._signal_bus.register_listener(BgpSignalBus.BGP_VRF_STATS_CONFIG_CHANGED, lambda _, vrf_conf: self.on_stats_config_change(vrf_conf))

    @property
    def router_id(self):
        return self._common_config.router_id

    @property
    def global_interested_rts(self):
        return self._rt_mgr.global_interested_rts

    @property
    def asn(self):
        return self._asn

    @property
    def table_manager(self):
        return self._table_manager

    @property
    def importmap_manager(self):
        return self._importmap_manager

    @property
    def peer_manager(self):
        return self._peer_manager

    @property
    def rt_manager(self):
        return self._rt_mgr

    @property
    def signal_bus(self):
        return self._signal_bus

    def enqueue_for_bgp_processing(self, dest):
        return self._bgp_processor.enqueue(dest)

    def on_vrf_removed(self, route_dist):
        vrf_stats_timer = self._timers.get(route_dist)
        if vrf_stats_timer:
            vrf_stats_timer.stop()
            del self._timers[route_dist]

    def on_vrf_added(self, vrf_conf):
        rd = vrf_conf.route_dist
        rf = vrf_conf.route_family
        vrf_table = self._table_manager.get_vrf_table(rd, rf)
        vrf_stats_timer = self._create_timer(rd, stats.log, stats_source=vrf_table.get_stats_summary_dict)
        if vrf_conf.stats_log_enabled:
            vrf_stats_timer.start(vrf_conf.stats_time)

    def on_stats_config_change(self, vrf_conf):
        vrf_stats_timer = self._timers.get(vrf_conf.route_dist)
        vrf_stats_timer.stop()
        vrf_stats_timer.start(vrf_conf.stats_time)

    def _run(self, *args, **kwargs):
        from os_ken.services.protocols.bgp.processor import BgpProcessor
        self._bgp_processor = BgpProcessor(self)
        processor_thread = self._spawn_activity(self._bgp_processor)
        for peer in self._peer_manager.iterpeers:
            self._spawn_activity(peer, self.start_protocol)
        waiter = kwargs.pop('waiter')
        waiter.set()
        self.listen_sockets = {}
        if self._common_config.bgp_server_port != 0:
            for host in self._common_config.bgp_server_hosts:
                server_thread, sockets = self._listen_tcp((host, self._common_config.bgp_server_port), self.start_protocol)
                self.listen_sockets.update(sockets)
                server_thread.wait()
        processor_thread.wait()

    def update_rtfilters(self):
        """Updates RT filters for each peer.

        Should be called if a new RT Nlri's have changed based on the setting.
        Currently only used by `Processor` to update the RT filters after it
        has processed a RT destination. If RT filter has changed for a peer we
        call RT filter change handler.
        """
        new_peer_to_rtfilter_map = self._compute_rtfilter_map()
        for peer in self._peer_manager.iterpeers:
            pre_rt_filter = self._rt_mgr.peer_to_rtfilter_map.get(peer, set())
            curr_rt_filter = new_peer_to_rtfilter_map.get(peer, set())
            old_rts = pre_rt_filter - curr_rt_filter
            new_rts = curr_rt_filter - pre_rt_filter
            if new_rts or old_rts:
                LOG.debug('RT Filter for peer %s updated: Added RTs %s, Removed Rts %s', peer.ip_address, new_rts, old_rts)
                self._on_update_rt_filter(peer, new_rts, old_rts)
        self._peer_manager.set_peer_to_rtfilter_map(new_peer_to_rtfilter_map)
        self._rt_mgr.peer_to_rtfilter_map = new_peer_to_rtfilter_map
        LOG.debug('Updated RT filters: %s', self._rt_mgr.peer_to_rtfilter_map)
        self._rt_mgr.update_interested_rts()

    def _on_update_rt_filter(self, peer, new_rts, old_rts):
        """Handles update of peer RT filter.

        Parameters:
            - `peer`: (Peer) whose RT filter has changed.
            - `new_rts`: (set) of new RTs that peer is interested in.
            - `old_rts`: (set) of RTs that peers is no longer interested in.
        """
        for table in self._table_manager._global_tables.values():
            if table.route_family == RF_RTC_UC:
                continue
            self._spawn('rt_filter_chg_%s' % peer, self._rt_mgr.on_rt_filter_chg_sync_peer, peer, new_rts, old_rts, table)
            LOG.debug('RT Filter change handler launched for route_family %s', table.route_family)

    def _compute_rtfilter_map(self):
        """Returns neighbor's RT filter (permit/allow filter based on RT).

        Walks RT filter tree and computes current RT filters for each peer that
        have advertised RT NLRIs.
        Returns:
            dict of peer, and `set` of rts that a particular neighbor is
            interested in.
        """
        rtfilter_map = {}

        def get_neigh_filter(neigh):
            neigh_filter = rtfilter_map.get(neigh)
            if neigh_filter is None:
                neigh_filter = set()
                rtfilter_map[neigh] = neigh_filter
            return neigh_filter
        if self._common_config.max_path_ext_rtfilter_all:
            for rtcdest in self._table_manager.get_rtc_table().values():
                known_path_list = rtcdest.known_path_list
                for path in known_path_list:
                    neigh = path.source
                    if neigh is None:
                        continue
                    neigh_filter = get_neigh_filter(neigh)
                    neigh_filter.add(path.nlri.route_target)
        else:
            for rtcdest in self._table_manager.get_rtc_table().values():
                path = rtcdest.best_path
                if not path:
                    continue
                neigh = path.source
                if neigh and neigh.is_ebgp_peer():
                    neigh_filter = get_neigh_filter(neigh)
                    neigh_filter.add(path.nlri.route_target)
                else:
                    known_path_list = rtcdest.known_path_list
                    for path in known_path_list:
                        neigh = path.source
                        if neigh and (not neigh.is_ebgp_peer()):
                            neigh_filter = get_neigh_filter(neigh)
                            neigh_filter.add(path.nlri.route_target)
        return rtfilter_map

    def register_flexinet_sink(self, sink):
        self._sinks.add(sink)

    def unregister_flexinet_sink(self, sink):
        self._sinks.remove(sink)

    def update_flexinet_peers(self, path, route_dist):
        for sink in self._sinks:
            out_route = FlexinetOutgoingRoute(path, route_dist)
            sink.enque_outgoing_msg(out_route)

    def _set_password(self, address, password):
        if ip.valid_ipv4(address):
            family = socket.AF_INET
        else:
            family = socket.AF_INET6
        for sock in self.listen_sockets.values():
            if sock.family == family:
                sockopt.set_tcp_md5sig(sock, address, password)

    def on_peer_added(self, peer):
        if peer._neigh_conf.password:
            self._set_password(peer._neigh_conf.ip_address, peer._neigh_conf.password)
        if self.started:
            self._spawn_activity(peer, self.start_protocol)
        if peer.rtc_as != self.asn:
            self._spawn('NEW_RTC_AS_HANDLER %s' % peer.rtc_as, self._rt_mgr.update_rtc_as_set)

    def on_peer_removed(self, peer):
        if peer._neigh_conf.password:
            self._set_password(peer._neigh_conf.ip_address, '')
        if peer.rtc_as != self.asn:
            self._spawn('OLD_RTC_AS_HANDLER %s' % peer.rtc_as, self._rt_mgr.update_rtc_as_set)

    def build_protocol(self, socket):
        assert socket
        _, remote_port = self.get_remotename(socket)
        remote_port = int(remote_port)
        is_reactive_conn = True
        if remote_port == STD_BGP_SERVER_PORT_NUM:
            is_reactive_conn = False
        bgp_protocol = self.protocol(socket, self._signal_bus, is_reactive_conn=is_reactive_conn)
        return bgp_protocol

    def start_protocol(self, socket):
        """Handler of new connection requests on bgp server port.

        Checks if new connection request is valid and starts new instance of
        protocol.
        """
        assert socket
        peer_addr, peer_port = self.get_remotename(socket)
        peer = self._peer_manager.get_by_addr(peer_addr)
        bgp_proto = self.build_protocol(socket)
        if not peer or not peer.enabled:
            LOG.debug('Closed connection %s %s:%s as it is not a recognized peer.', 'from' if bgp_proto.is_reactive else 'to', peer_addr, peer_port)
            code = BGP_ERROR_CEASE
            subcode = BGP_ERROR_SUB_CONNECTION_RESET
            bgp_proto.send_notification(code, subcode)
        elif bgp_proto.is_reactive and peer.connect_mode is CONNECT_MODE_ACTIVE:
            LOG.debug('Closed connection from %s:%s as connect_mode is configured ACTIVE.', peer_addr, peer_port)
            code = BGP_ERROR_CEASE
            subcode = BGP_ERROR_SUB_CONNECTION_RESET
            bgp_proto.send_notification(code, subcode)
        elif not (peer.in_idle() or peer.in_active() or peer.in_connect()):
            LOG.debug('Closing connection to %s:%s as we have connection in state other than IDLE or ACTIVE, i.e. connection resolution', peer_addr, peer_port)
            code = BGP_ERROR_CEASE
            subcode = BGP_ERROR_SUB_CONNECTION_COLLISION_RESOLUTION
            bgp_proto.send_notification(code, subcode)
        else:
            bind_ip, bind_port = self.get_localname(socket)
            peer._host_bind_ip = bind_ip
            peer._host_bind_port = bind_port
            self._spawn_activity(bgp_proto, peer)

    def start_bmp(self, host, port):
        if (host, port) in self.bmpclients:
            bmpclient = self.bmpclients[host, port]
            if bmpclient.started:
                LOG.warning('bmpclient is already running for %s:%s', host, port)
                return False
        bmpclient = BMPClient(self, host, port)
        self.bmpclients[host, port] = bmpclient
        self._spawn_activity(bmpclient)
        return True

    def stop_bmp(self, host, port):
        if (host, port) not in self.bmpclients:
            LOG.warning('no bmpclient is running for %s:%s', host, port)
            return False
        bmpclient = self.bmpclients[host, port]
        bmpclient.stop()