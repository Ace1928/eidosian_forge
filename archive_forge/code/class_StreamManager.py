from binascii import hexlify
from hashlib import sha1
from sys import intern
from typing import Optional, Tuple
from zope.interface import directlyProvides, implementer
from twisted.internet import defer, protocol
from twisted.internet.error import ConnectionLost
from twisted.python import failure, log, randbytes
from twisted.words.protocols.jabber import error, ijabber, jid
from twisted.words.xish import domish, xmlstream
from twisted.words.xish.xmlstream import (
class StreamManager(XMPPHandlerCollection):
    """
    Business logic representing a managed XMPP connection.

    This maintains a single XMPP connection and provides facilities for packet
    routing and transmission. Business logic modules are objects providing
    L{ijabber.IXMPPHandler} (like subclasses of L{XMPPHandler}), and added
    using L{addHandler}.

    @ivar xmlstream: currently managed XML stream
    @type xmlstream: L{XmlStream}
    @ivar logTraffic: if true, log all traffic.
    @type logTraffic: C{bool}
    @ivar _initialized: Whether the stream represented by L{xmlstream} has
                        been initialized. This is used when caching outgoing
                        stanzas.
    @type _initialized: C{bool}
    @ivar _packetQueue: internal buffer of unsent data. See L{send} for details.
    @type _packetQueue: C{list}
    """
    logTraffic = False

    def __init__(self, factory):
        XMPPHandlerCollection.__init__(self)
        self.xmlstream = None
        self._packetQueue = []
        self._initialized = False
        factory.addBootstrap(STREAM_CONNECTED_EVENT, self._connected)
        factory.addBootstrap(STREAM_AUTHD_EVENT, self._authd)
        factory.addBootstrap(INIT_FAILED_EVENT, self.initializationFailed)
        factory.addBootstrap(STREAM_END_EVENT, self._disconnected)
        self.factory = factory

    def addHandler(self, handler):
        """
        Add protocol handler.

        When an XML stream has already been established, the handler's
        C{connectionInitialized} will be called to get it up to speed.
        """
        XMPPHandlerCollection.addHandler(self, handler)
        if self.xmlstream and self._initialized:
            handler.makeConnection(self.xmlstream)
            handler.connectionInitialized()

    def _connected(self, xs):
        """
        Called when the transport connection has been established.

        Here we optionally set up traffic logging (depending on L{logTraffic})
        and call each handler's C{makeConnection} method with the L{XmlStream}
        instance.
        """

        def logDataIn(buf):
            log.msg('RECV: %r' % buf)

        def logDataOut(buf):
            log.msg('SEND: %r' % buf)
        if self.logTraffic:
            xs.rawDataInFn = logDataIn
            xs.rawDataOutFn = logDataOut
        self.xmlstream = xs
        for e in self:
            e.makeConnection(xs)

    def _authd(self, xs):
        """
        Called when the stream has been initialized.

        Send out cached stanzas and call each handler's
        C{connectionInitialized} method.
        """
        for p in self._packetQueue:
            xs.send(p)
        self._packetQueue = []
        self._initialized = True
        for e in self:
            e.connectionInitialized()

    def initializationFailed(self, reason):
        """
        Called when stream initialization has failed.

        Stream initialization has halted, with the reason indicated by
        C{reason}. It may be retried by calling the authenticator's
        C{initializeStream}. See the respective authenticators for details.

        @param reason: A failure instance indicating why stream initialization
                       failed.
        @type reason: L{failure.Failure}
        """

    def _disconnected(self, reason):
        """
        Called when the stream has been closed.

        From this point on, the manager doesn't interact with the
        L{XmlStream} anymore and notifies each handler that the connection
        was lost by calling its C{connectionLost} method.
        """
        self.xmlstream = None
        self._initialized = False
        for e in self:
            e.connectionLost(reason)

    def send(self, obj):
        """
        Send data over the XML stream.

        When there is no established XML stream, the data is queued and sent
        out when a new XML stream has been established and initialized.

        @param obj: data to be sent over the XML stream. See
                    L{xmlstream.XmlStream.send} for details.
        """
        if self._initialized:
            self.xmlstream.send(obj)
        else:
            self._packetQueue.append(obj)