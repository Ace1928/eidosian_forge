import random
from typing import Any, Callable, Optional
from zope.interface import implementer
from twisted.internet import defer, error, interfaces
from twisted.internet.interfaces import IAddress, ITransport
from twisted.logger import _loggerFor
from twisted.python import components, failure, log
class AbstractDatagramProtocol:
    """
    Abstract protocol for datagram-oriented transports, e.g. IP, ICMP, ARP,
    UDP.
    """
    transport = None
    numPorts = 0
    noisy = True

    def __getstate__(self):
        d = self.__dict__.copy()
        d['transport'] = None
        return d

    def doStart(self):
        """
        Make sure startProtocol is called.

        This will be called by makeConnection(), users should not call it.
        """
        if not self.numPorts:
            if self.noisy:
                log.msg('Starting protocol %s' % self)
            self.startProtocol()
        self.numPorts = self.numPorts + 1

    def doStop(self):
        """
        Make sure stopProtocol is called.

        This will be called by the port, users should not call it.
        """
        assert self.numPorts > 0
        self.numPorts = self.numPorts - 1
        self.transport = None
        if not self.numPorts:
            if self.noisy:
                log.msg('Stopping protocol %s' % self)
            self.stopProtocol()

    def startProtocol(self):
        """
        Called when a transport is connected to this protocol.

        Will only be called once, even if multiple ports are connected.
        """

    def stopProtocol(self):
        """
        Called when the transport is disconnected.

        Will only be called once, after all ports are disconnected.
        """

    def makeConnection(self, transport):
        """
        Make a connection to a transport and a server.

        This sets the 'transport' attribute of this DatagramProtocol, and calls the
        doStart() callback.
        """
        assert self.transport == None
        self.transport = transport
        self.doStart()

    def datagramReceived(self, datagram: bytes, addr: Any) -> None:
        """
        Called when a datagram is received.

        @param datagram: the bytes received from the transport.
        @param addr: tuple of source of datagram.
        """