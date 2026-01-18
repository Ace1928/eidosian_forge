from os_ken.services.protocols.bgp.base import Activity
from os_ken.lib import hub
from os_ken.lib.packet import bmp
from os_ken.lib.packet import bgp
import socket
import logging
from calendar import timegm
from os_ken.services.protocols.bgp.signals.emit import BgpSignalBus
from os_ken.services.protocols.bgp.info_base.ipv4 import Ipv4Path
from os_ken.lib.packet.bgp import BGPUpdate
from os_ken.lib.packet.bgp import BGPPathAttributeMpUnreachNLRI
def _handle_bmp_session(self, socket):
    self._socket = socket
    init_info = {'type': bmp.BMP_INIT_TYPE_STRING, 'value': 'This is OSKen BGP BMP message'}
    init_msg = bmp.BMPInitiation([init_info])
    self._send(init_msg)
    peer_manager = self._core_service.peer_manager
    for peer in (p for p in peer_manager.iterpeers if p.in_established()):
        msg = self._construct_peer_up_notification(peer)
        self._send(msg)
        for path in peer._adj_rib_in.values():
            msg = self._construct_route_monitoring(peer, path)
            self._send(msg)
    while True:
        ret = self._socket.recv(1)
        if len(ret) == 0:
            LOG.debug('BMP socket is closed. retry connecting..')
            self._socket = None
            self._connect_retry_event.set()
            break