import contextlib
import socket
import struct
from os_ken.controller import handler
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.lib import addrconv
from os_ken.lib import hub
from os_ken.lib.packet import arp
from os_ken.lib.packet import vrrp
from os_ken.services.protocols.vrrp import monitor
from os_ken.services.protocols.vrrp import event as vrrp_event
from os_ken.services.protocols.vrrp import utils
def _join_vrrp_group(self, join_leave):
    if join_leave:
        join_leave = MCAST_JOIN_GROUP
    else:
        join_leave = MCAST_LEAVE_GROUP
    group_req = struct.pack('I', self.ifindex)
    group_req += b'\x00' * (struct.calcsize('P') - struct.calcsize('I'))
    if self.config.is_ipv6:
        family = socket.IPPROTO_IPV6
        sockaddr = struct.pack('H', socket.AF_INET6)
        sockaddr += struct.pack('!H', 0)
        sockaddr += struct.pack('!I', 0)
        sockaddr += addrconv.ipv6.text_to_bin(vrrp.VRRP_IPV6_DST_ADDRESS)
        sockaddr += struct.pack('I', 0)
    else:
        family = socket.IPPROTO_IP
        sockaddr = struct.pack('H', socket.AF_INET)
        sockaddr += struct.pack('!H', 0)
        sockaddr += addrconv.ipv4.text_to_bin(vrrp.VRRP_IPV4_DST_ADDRESS)
    sockaddr += b'\x00' * (SS_MAXSIZE - len(sockaddr))
    group_req += sockaddr
    self.ip_socket.setsockopt(family, join_leave, group_req)
    return