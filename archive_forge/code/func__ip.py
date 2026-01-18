import socket
import struct
from collections import deque
from errno import EAGAIN, EBADF, EINTR, EINVAL, ENOBUFS, ENOSYS, EPERM, EWOULDBLOCK
from functools import wraps
from zope.interface import implementer
from twisted.internet.protocol import DatagramProtocol
from twisted.pair.ethernet import EthernetProtocol
from twisted.pair.ip import IPProtocol
from twisted.pair.rawudp import RawUDPProtocol
from twisted.pair.tuntap import _IFNAMSIZ, _TUNSETIFF, TunnelFlags, _IInputOutputSystem
from twisted.python.compat import nativeString
def _ip(src, dst, payload):
    """
    Construct an IP datagram with the given source, destination, and
    application payload.

    @param src: The source IPv4 address as a dotted-quad string.
    @type src: L{bytes}

    @param dst: The destination IPv4 address as a dotted-quad string.
    @type dst: L{bytes}

    @param payload: The content of the IP datagram (such as a UDP datagram).
    @type payload: L{bytes}

    @return: An IP datagram header and payload.
    @rtype: L{bytes}
    """
    ipHeader = b'E\x00' + _H(20 + len(payload)) + b'\x00\x01\x00\x00@\x11' + _H(0) + socket.inet_pton(socket.AF_INET, nativeString(src)) + socket.inet_pton(socket.AF_INET, nativeString(dst))
    checksumStep1 = sum(struct.unpack('!10H', ipHeader))
    carry = checksumStep1 >> 16
    checksumStep2 = (checksumStep1 & 65535) + carry
    checksumStep3 = checksumStep2 ^ 65535
    ipHeader = ipHeader[:10] + struct.pack('!H', checksumStep3) + ipHeader[12:]
    return ipHeader + payload