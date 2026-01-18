import abc
from collections import OrderedDict
import logging
import socket
import time
import traceback
import weakref
import netaddr
from os_ken.lib import hub
from os_ken.lib import sockopt
from os_ken.lib import ip
from os_ken.lib.hub import Timeout
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
from os_ken.services.protocols.bgp.utils.circlist import CircularListType
from os_ken.services.protocols.bgp.utils.evtlet import LoopingCall
def _connect_tcp(self, peer_addr, conn_handler, time_out=None, bind_address=None, password=None):
    """Creates a TCP connection to given peer address.

        Tries to create a socket for `timeout` number of seconds. If
        successful, uses the socket instance to start `client_factory`.
        The socket is bound to `bind_address` if specified.
        """
    LOG.debug('Connect TCP called for %s:%s', peer_addr[0], peer_addr[1])
    if ip.valid_ipv4(peer_addr[0]):
        family = socket.AF_INET
    else:
        family = socket.AF_INET6
    with Timeout(time_out, socket.error):
        sock = socket.socket(family)
        if bind_address:
            sock.bind(bind_address)
        if password:
            sockopt.set_tcp_md5sig(sock, peer_addr[0], password)
        sock.connect(peer_addr)
    local = self.get_localname(sock)[0]
    remote = self.get_remotename(sock)[0]
    conn_name = 'L: ' + local + ', R: ' + remote
    self._asso_socket_map[conn_name] = sock
    self._spawn(conn_name, conn_handler, sock)
    return sock