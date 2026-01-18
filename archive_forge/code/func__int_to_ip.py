import uuid
import socket
import struct
from libcloud.common.base import ConnectionKey
from libcloud.compute.base import Node, KeyPair, NodeSize, NodeImage, NodeDriver, NodeLocation
from libcloud.compute.types import Provider, NodeState
def _int_to_ip(ip):
    return socket.inet_ntoa(struct.pack('I', socket.ntohl(ip)))