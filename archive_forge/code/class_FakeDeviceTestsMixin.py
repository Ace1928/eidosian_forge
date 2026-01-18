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
class FakeDeviceTestsMixin:
    """
    Define a mixin for use with test cases that require an
    L{_IInputOutputSystem} provider.  This mixin hands out L{MemoryIOSystem}
    instances as the provider of that interface.
    """
    _TUNNEL_DEVICE = b'tap-twistedtest'
    _TUNNEL_LOCAL = b'172.16.2.1'
    _TUNNEL_REMOTE = b'172.16.2.2'

    def createSystem(self):
        """
        Create and return a brand new L{MemoryIOSystem}.

        The L{MemoryIOSystem} knows how to open new tunnel devices.

        @return: The newly created I/O system object.
        @rtype: L{MemoryIOSystem}
        """
        system = MemoryIOSystem()
        system.registerSpecialDevice(Tunnel._DEVICE_NAME, Tunnel)
        return system