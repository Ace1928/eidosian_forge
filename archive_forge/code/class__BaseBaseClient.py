from __future__ import annotations
import os
import socket
import struct
import sys
from typing import Callable, ClassVar, List, Optional, Union
from zope.interface import Interface, implementer
import attr
import typing_extensions
from twisted.internet.interfaces import (
from twisted.logger import ILogObserver, LogEvent, Logger
from twisted.python import deprecate, versions
from twisted.python.compat import lazyByteSlice
from twisted.python.runtime import platformType
from errno import errorcode
from twisted.internet import abstract, address, base, error, fdesc, main
from twisted.internet.error import CannotListenError
from twisted.internet.protocol import Protocol
from twisted.internet.task import deferLater
from twisted.python import failure, log, reflect
from twisted.python.util import untilConcludes
class _BaseBaseClient:
    """
    Code shared with other (non-POSIX) reactors for management of general
    outgoing connections.

    Requirements upon subclasses are documented as instance variables rather
    than abstract methods, in order to avoid MRO confusion, since this base is
    mixed in to unfortunately weird and distinctive multiple-inheritance
    hierarchies and many of these attributes are provided by peer classes
    rather than descendant classes in those hierarchies.

    @ivar addressFamily: The address family constant (C{socket.AF_INET},
        C{socket.AF_INET6}, C{socket.AF_UNIX}) of the underlying socket of this
        client connection.
    @type addressFamily: C{int}

    @ivar socketType: The socket type constant (C{socket.SOCK_STREAM} or
        C{socket.SOCK_DGRAM}) of the underlying socket.
    @type socketType: C{int}

    @ivar _requiresResolution: A flag indicating whether the address of this
        client will require name resolution.  C{True} if the hostname of said
        address indicates a name that must be resolved by hostname lookup,
        C{False} if it indicates an IP address literal.
    @type _requiresResolution: C{bool}

    @cvar _commonConnection: Subclasses must provide this attribute, which
        indicates the L{Connection}-alike class to invoke C{__init__} and
        C{connectionLost} on.
    @type _commonConnection: C{type}

    @ivar _stopReadingAndWriting: Subclasses must implement in order to remove
        this transport from its reactor's notifications in response to a
        terminated connection attempt.
    @type _stopReadingAndWriting: 0-argument callable returning L{None}

    @ivar _closeSocket: Subclasses must implement in order to close the socket
        in response to a terminated connection attempt.
    @type _closeSocket: 1-argument callable; see L{_SocketCloser._closeSocket}

    @ivar _collectSocketDetails: Clean up references to the attached socket in
        its underlying OS resource (such as a file descriptor or file handle),
        as part of post connection-failure cleanup.
    @type _collectSocketDetails: 0-argument callable returning L{None}.

    @ivar reactor: The class pointed to by C{_commonConnection} should set this
        attribute in its constructor.
    @type reactor: L{twisted.internet.interfaces.IReactorTime},
        L{twisted.internet.interfaces.IReactorCore},
        L{twisted.internet.interfaces.IReactorFDSet}
    """
    addressFamily = socket.AF_INET
    socketType = socket.SOCK_STREAM

    def _finishInit(self, whenDone, skt, error, reactor):
        """
        Called by subclasses to continue to the stage of initialization where
        the socket connect attempt is made.

        @param whenDone: A 0-argument callable to invoke once the connection is
            set up.  This is L{None} if the connection could not be prepared
            due to a previous error.

        @param skt: The socket object to use to perform the connection.
        @type skt: C{socket._socketobject}

        @param error: The error to fail the connection with.

        @param reactor: The reactor to use for this client.
        @type reactor: L{twisted.internet.interfaces.IReactorTime}
        """
        if whenDone:
            self._commonConnection.__init__(self, skt, None, reactor)
            reactor.callLater(0, whenDone)
        else:
            reactor.callLater(0, self.failIfNotConnected, error)

    def resolveAddress(self):
        """
        Resolve the name that was passed to this L{_BaseBaseClient}, if
        necessary, and then move on to attempting the connection once an
        address has been determined.  (The connection will be attempted
        immediately within this function if either name resolution can be
        synchronous or the address was an IP address literal.)

        @note: You don't want to call this method from outside, as it won't do
            anything useful; it's just part of the connection bootstrapping
            process.  Also, although this method is on L{_BaseBaseClient} for
            historical reasons, it's not used anywhere except for L{Client}
            itself.

        @return: L{None}
        """
        if self._requiresResolution:
            d = self.reactor.resolve(self.addr[0])
            d.addCallback(lambda n: (n,) + self.addr[1:])
            d.addCallbacks(self._setRealAddress, self.failIfNotConnected)
        else:
            self._setRealAddress(self.addr)

    def _setRealAddress(self, address):
        """
        Set the resolved address of this L{_BaseBaseClient} and initiate the
        connection attempt.

        @param address: Depending on whether this is an IPv4 or IPv6 connection
            attempt, a 2-tuple of C{(host, port)} or a 4-tuple of C{(host,
            port, flow, scope)}.  At this point it is a fully resolved address,
            and the 'host' portion will always be an IP address, not a DNS
            name.
        """
        if len(address) == 4:
            hostname = socket.getnameinfo(address, socket.NI_NUMERICHOST | socket.NI_NUMERICSERV)[0]
            self.realAddress = tuple([hostname] + list(address[1:]))
        else:
            self.realAddress = address
        self.doConnect()

    def failIfNotConnected(self, err):
        """
        Generic method called when the attempts to connect failed. It basically
        cleans everything it can: call connectionFailed, stop read and write,
        delete socket related members.
        """
        if self.connected or self.disconnected or (not hasattr(self, 'connector')):
            return
        self._stopReadingAndWriting()
        try:
            self._closeSocket(True)
        except AttributeError:
            pass
        else:
            self._collectSocketDetails()
        self.connector.connectionFailed(failure.Failure(err))
        del self.connector

    def stopConnecting(self):
        """
        If a connection attempt is still outstanding (i.e.  no connection is
        yet established), immediately stop attempting to connect.
        """
        self.failIfNotConnected(error.UserError())

    def connectionLost(self, reason):
        """
        Invoked by lower-level logic when it's time to clean the socket up.
        Depending on the state of the connection, either inform the attached
        L{Connector} that the connection attempt has failed, or inform the
        connected L{IProtocol} that the established connection has been lost.

        @param reason: the reason that the connection was terminated
        @type reason: L{Failure}
        """
        if not self.connected:
            self.failIfNotConnected(error.ConnectError(string=reason))
        else:
            self._commonConnection.connectionLost(self, reason)
            self.connector.connectionLost(reason)