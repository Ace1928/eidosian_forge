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
@implementer(_IInputOutputSystem)
class TestRealSystem(_RealSystem):
    """
    Add extra skipping logic so tests that try to create real tunnel devices on
    platforms where those are not supported automatically get skipped.
    """

    def open(self, filename, *args, **kwargs):
        """
        Attempt an open, but if the file is /dev/net/tun and it does not exist,
        translate the error into L{SkipTest} so that tests that require
        platform support for tuntap devices are skipped instead of failed.
        """
        try:
            return super().open(filename, *args, **kwargs)
        except OSError as e:
            if e.errno in (ENOENT, ENODEV) and filename == b'/dev/net/tun':
                raise SkipTest('Platform lacks /dev/net/tun')
            raise

    def ioctl(self, *args, **kwargs):
        """
        Attempt an ioctl, but translate permission denied errors into
        L{SkipTest} so that tests that require elevated system privileges and
        do not have them are skipped instead of failed.
        """
        try:
            return super().ioctl(*args, **kwargs)
        except OSError as e:
            if EPERM == e.errno:
                raise SkipTest('Permission to configure device denied')
            raise

    def sendUDP(self, datagram, address):
        """
        Use the platform network stack to send a datagram to the given address.

        @param datagram: A UDP datagram payload to send.
        @type datagram: L{bytes}

        @param address: The destination to which to send the datagram.
        @type address: L{tuple} of (L{bytes}, L{int})

        @return: The address from which the UDP datagram was sent.
        @rtype: L{tuple} of (L{bytes}, L{int})
        """
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind(('172.16.0.1', 0))
        s.sendto(datagram, address)
        return s.getsockname()

    def receiveUDP(self, fileno, host, port):
        """
        Use the platform network stack to receive a datagram sent to the given
        address.

        @param fileno: The file descriptor of the tunnel used to send the
            datagram.  This is ignored because a real socket is used to receive
            the datagram.
        @type fileno: L{int}

        @param host: The IPv4 address at which the datagram will be received.
        @type host: L{bytes}

        @param port: The UDP port number at which the datagram will be
            received.
        @type port: L{int}

        @return: A L{socket.socket} which can be used to receive the specified
            datagram.
        """
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind((host, port))
        return s