import os
import socket
import struct
from collections import deque
from errno import EAGAIN, EBADF, EINVAL, ENODEV, ENOENT, EPERM, EWOULDBLOCK
from itertools import cycle
from random import randrange
from signal import SIGINT
from typing import Optional
from twisted.python.reflect import ObjectNotFound, namedAny
from zope.interface import Interface, implementer
from zope.interface.verify import verifyObject
from twisted.internet.error import CannotListenError
from twisted.internet.interfaces import IAddress, IListeningPort, IReactorFDSet
from twisted.internet.protocol import (
from twisted.internet.task import Clock
from twisted.pair.ethernet import EthernetProtocol
from twisted.pair.ip import IPProtocol
from twisted.pair.raw import IRawPacketProtocol
from twisted.pair.rawudp import RawUDPProtocol
from twisted.python.compat import iterbytes
from twisted.python.log import addObserver, removeObserver, textFromEventDict
from twisted.python.reflect import fullyQualifiedName
from twisted.trial.unittest import SkipTest, SynchronousTestCase
class TapHelper:
    """
    A helper for tests of tap-related functionality (ethernet-level tunnels).
    """

    @property
    def TUNNEL_TYPE(self):
        flag = TunnelFlags.IFF_TAP
        if not self.pi:
            flag |= TunnelFlags.IFF_NO_PI
        return flag

    def __init__(self, tunnelRemote, tunnelLocal, pi):
        """
        @param tunnelRemote: The source address for UDP datagrams originated
            from this helper.  This is an IPv4 dotted-quad string.
        @type tunnelRemote: L{bytes}

        @param tunnelLocal: The destination address for UDP datagrams
            originated from this helper.  This is an IPv4 dotted-quad string.
        @type tunnelLocal: L{bytes}

        @param pi: A flag indicating whether this helper will generate and
            consume a protocol information (PI) header.
        @type pi: L{bool}
        """
        self.tunnelRemote = tunnelRemote
        self.tunnelLocal = tunnelLocal
        self.pi = pi

    def encapsulate(self, source, destination, payload):
        """
        Construct an ethernet frame containing an ip datagram containing a udp
        datagram containing the given application-level payload.

        @param source: The source port for the UDP datagram being encapsulated.
        @type source: L{int}

        @param destination: The destination port for the UDP datagram being
            encapsulated.
        @type destination: L{int}

        @param payload: The application data to include in the udp datagram.
        @type payload: L{bytes}

        @return: An ethernet frame.
        @rtype: L{bytes}
        """
        tun = TunHelper(self.tunnelRemote, self.tunnelLocal)
        ip = tun.encapsulate(source, destination, payload)
        frame = _ethernet(src=b'\x00\x00\x00\x00\x00\x00', dst=b'\xff\xff\xff\xff\xff\xff', protocol=_IPv4, payload=ip)
        if self.pi:
            protocol = _IPv4
            flags = 0
            frame = _H(flags) + _H(protocol) + frame
        return frame

    def parser(self):
        """
        Get a function for parsing a datagram read from a I{tap} device.

        @return: A function which accepts a datagram exactly as might be read
            from a I{tap} device.  The datagram is expected to ultimately carry
            a UDP datagram.  When called, it returns a L{list} of L{tuple}s.
            Each tuple has the UDP application data as the first element and
            the sender address as the second element.
        """
        datagrams = []
        receiver = DatagramProtocol()

        def capture(*args):
            datagrams.append(args)
        receiver.datagramReceived = capture
        udp = RawUDPProtocol()
        udp.addProto(12345, receiver)
        ip = IPProtocol()
        ip.addProto(17, udp)
        ether = EthernetProtocol()
        ether.addProto(2048, ip)

        def parser(datagram):
            if self.pi:
                datagram = datagram[_PI_SIZE:]
            ether.datagramReceived(datagram)
            return datagrams
        return parser