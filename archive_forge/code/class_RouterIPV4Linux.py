import contextlib
import greenlet
import socket
from os_ken.base import app_manager
from os_ken.controller import handler
from os_ken.controller import ofp_event
from os_ken.lib import hub
from os_ken.lib import mac as mac_lib
from os_ken.lib.packet import arp
from os_ken.lib.packet import ethernet
from os_ken.lib.packet import packet
from os_ken.lib.packet import vlan
from os_ken.lib.packet import vrrp
from os_ken.ofproto import ether
from os_ken.ofproto import ofproto_v1_2
from os_ken.services.protocols.vrrp import api as vrrp_api
from os_ken.services.protocols.vrrp import event as vrrp_event
from os_ken.services.protocols.vrrp import utils
class RouterIPV4Linux(RouterIPV4):

    def __init__(self, *args, **kwargs):
        super(RouterIPV4Linux, self).__init__(*args, **kwargs)
        assert isinstance(self.interface, vrrp_event.VRRPInterfaceNetworkDevice)
        self.__is_master = False
        self._arp_thread = None

    def start(self):
        self._disable_router()
        super(RouterIPV4Linux, self).start()

    def _initialized_to_master(self):
        self.logger.debug('initialized to master')
        self._master()

    def _become_master(self):
        self.logger.debug('become master')
        self._master()

    def _master(self):
        self.__is_master = True
        self._enable_router()
        self._send_garp()

    def _become_backup(self):
        self.logger.debug('become backup')
        self.__is_master = False
        self._disable_router()

    def _shutdowned(self):
        self._disable_router()

    def _arp_loop_socket(self, packet_socket):
        while True:
            try:
                buf = packet_socket.recv(1500)
            except socket.timeout:
                continue
            self._arp_process(buf)

    def _arp_loop(self):
        try:
            with contextlib.closing(socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(ether.ETH_TYPE_ARP))) as packet_socket:
                packet_socket.bind((self.interface.device_name, socket.htons(ether.ETH_TYPE_ARP), socket.PACKET_BROADCAST, arp.ARP_HW_TYPE_ETHERNET, mac_lib.BROADCAST))
                self._arp_loop_socket(packet_socket)
        except greenlet.GreenletExit:
            pass

    def _enable_router(self):
        if self._arp_thread is None:
            self._arp_thread = hub.spawn(self._arp_loop)
        self.logger.debug('TODO:_enable_router')

    def _disable_router(self):
        if self._arp_thread is not None:
            self._arp_thread.kill()
            hub.joinall([self._arp_thread])
            self._arp_thread = None
        self.logger.debug('TODO:_disable_router')