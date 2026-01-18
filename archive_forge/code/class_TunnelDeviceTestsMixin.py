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
class TunnelDeviceTestsMixin:
    """
    A mixin defining tests that apply to L{_IInputOutputSystem}
    implementations.
    """

    def setUp(self):
        """
        Create the L{_IInputOutputSystem} provider under test and open a tunnel
        using it.
        """
        self.system = self.createSystem()
        self.fileno = self.system.open(b'/dev/net/tun', os.O_RDWR | os.O_NONBLOCK)
        self.addCleanup(self.system.close, self.fileno)
        mode = self.helper.TUNNEL_TYPE
        config = struct.pack('%dsH' % (_IFNAMSIZ,), self._TUNNEL_DEVICE, mode.value)
        self.system.ioctl(self.fileno, _TUNSETIFF, config)

    def test_interface(self):
        """
        The object under test provides L{_IInputOutputSystem}.
        """
        self.assertTrue(verifyObject(_IInputOutputSystem, self.system))

    def _invalidFileDescriptor(self):
        """
        Get an invalid file descriptor.

        @return: An integer which is not a valid file descriptor at the time of
            this call.  After any future system call which allocates a new file
            descriptor, there is no guarantee the returned file descriptor will
            still be invalid.
        """
        fd = self.system.open(b'/dev/net/tun', os.O_RDWR)
        self.system.close(fd)
        return fd

    def test_readEBADF(self):
        """
        The device's C{read} implementation raises L{OSError} with an errno of
        C{EBADF} when called on a file descriptor which is not valid (ie, which
        has no associated file description).
        """
        fd = self._invalidFileDescriptor()
        exc = self.assertRaises(OSError, self.system.read, fd, 1024)
        self.assertEqual(EBADF, exc.errno)

    def test_writeEBADF(self):
        """
        The device's C{write} implementation raises L{OSError} with an errno of
        C{EBADF} when called on a file descriptor which is not valid (ie, which
        has no associated file description).
        """
        fd = self._invalidFileDescriptor()
        exc = self.assertRaises(OSError, self.system.write, fd, b'bytes')
        self.assertEqual(EBADF, exc.errno)

    def test_closeEBADF(self):
        """
        The device's C{close} implementation raises L{OSError} with an errno of
        C{EBADF} when called on a file descriptor which is not valid (ie, which
        has no associated file description).
        """
        fd = self._invalidFileDescriptor()
        exc = self.assertRaises(OSError, self.system.close, fd)
        self.assertEqual(EBADF, exc.errno)

    def test_ioctlEBADF(self):
        """
        The device's C{ioctl} implementation raises L{OSError} with an errno of
        C{EBADF} when called on a file descriptor which is not valid (ie, which
        has no associated file description).
        """
        fd = self._invalidFileDescriptor()
        exc = self.assertRaises(IOError, self.system.ioctl, fd, _TUNSETIFF, b'tap0')
        self.assertEqual(EBADF, exc.errno)

    def test_ioctlEINVAL(self):
        """
        The device's C{ioctl} implementation raises L{IOError} with an errno of
        C{EINVAL} when called with a request (second argument) which is not a
        supported operation.
        """
        request = 3735928559
        exc = self.assertRaises(IOError, self.system.ioctl, self.fileno, request, b'garbage')
        self.assertEqual(EINVAL, exc.errno)

    def test_receive(self):
        """
        If a UDP datagram is sent to an address reachable by the tunnel device
        then it can be read out of the tunnel device.
        """
        parse = self.helper.parser()
        found = False
        for i in range(100):
            key = randrange(2 ** 64)
            message = b'hello world:%d' % (key,)
            source = self.system.sendUDP(message, (self._TUNNEL_REMOTE, 12345))
            for j in range(100):
                try:
                    packet = self.system.read(self.fileno, 1024)
                except OSError as e:
                    if e.errno in (EAGAIN, EWOULDBLOCK):
                        break
                    raise
                else:
                    datagrams = parse(packet)
                    if (message, source) in datagrams:
                        found = True
                        break
                    del datagrams[:]
            if found:
                break
        if not found:
            self.fail('Never saw probe UDP packet on tunnel')

    def test_send(self):
        """
        If a UDP datagram is written the tunnel device then it is received by
        the network to which it is addressed.
        """
        key = randrange(2 ** 64)
        message = b'hello world:%d' % (key,)
        self.addCleanup(socket.setdefaulttimeout, socket.getdefaulttimeout())
        socket.setdefaulttimeout(120)
        port = self.system.receiveUDP(self.fileno, self._TUNNEL_LOCAL, 12345)
        packet = self.helper.encapsulate(50000, 12345, message)
        self.system.write(self.fileno, packet)
        packet = port.recv(1024)
        self.assertEqual(message, packet)