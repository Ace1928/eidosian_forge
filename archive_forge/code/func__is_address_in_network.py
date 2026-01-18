import os
import socket
import struct
from six.moves.urllib.parse import urlparse
def _is_address_in_network(ip, net):
    ipaddr = struct.unpack('I', socket.inet_aton(ip))[0]
    netaddr, bits = net.split('/')
    netmask = struct.unpack('I', socket.inet_aton(netaddr))[0] & (2 << int(bits) - 1) - 1
    return ipaddr & netmask == netmask