import uuid
import socket
import struct
from libcloud.common.base import ConnectionKey
from libcloud.compute.base import Node, KeyPair, NodeSize, NodeImage, NodeDriver, NodeLocation
from libcloud.compute.types import Provider, NodeState
def _ip_to_int(ip):
    return socket.htonl(struct.unpack('I', socket.inet_aton(ip))[0])