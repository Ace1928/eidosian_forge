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
def _ethernet(src, dst, protocol, payload):
    """
    Construct an ethernet frame.

    @param src: The source ethernet address, encoded.
    @type src: L{bytes}

    @param dst: The destination ethernet address, encoded.
    @type dst: L{bytes}

    @param protocol: The protocol number of the payload of this datagram.
    @type protocol: L{int}

    @param payload: The content of the ethernet frame (such as an IP datagram).
    @type payload: L{bytes}

    @return: The full ethernet frame.
    @rtype: L{bytes}
    """
    return dst + src + _H(protocol) + payload