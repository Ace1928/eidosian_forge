import io
from collections import deque
from typing import List
from zope.interface import implementer
import h2.config
import h2.connection
import h2.errors
import h2.events
import h2.exceptions
import priority
from twisted.internet._producer_helpers import _PullToPush
from twisted.internet.defer import Deferred
from twisted.internet.error import ConnectionLost
from twisted.internet.interfaces import (
from twisted.internet.protocol import Protocol
from twisted.logger import Logger
from twisted.protocols.policies import TimeoutMixin
from twisted.python.failure import Failure
from twisted.web.error import ExcessiveBufferingError
@implementer(ITransport, IConsumer, IPushProducer)
class H2Stream:
    """
    A class representing a single HTTP/2 stream.

    This class works hand-in-hand with L{H2Connection}. It acts to provide an
    implementation of L{ITransport}, L{IConsumer}, and L{IProducer} that work
    for a single HTTP/2 connection, while tightly cleaving to the interface
    provided by those interfaces. It does this by having a tight coupling to
    L{H2Connection}, which allows associating many of the functions of
    L{ITransport}, L{IConsumer}, and L{IProducer} to objects on a
    stream-specific level.

    @ivar streamID: The numerical stream ID that this object corresponds to.
    @type streamID: L{int}

    @ivar producing: Whether this stream is currently allowed to produce data
        to its consumer.
    @type producing: L{bool}

    @ivar command: The HTTP verb used on the request.
    @type command: L{unicode}

    @ivar path: The HTTP path used on the request.
    @type path: L{unicode}

    @ivar producer: The object producing the response, if any.
    @type producer: L{IProducer}

    @ivar site: The L{twisted.web.server.Site} object this stream belongs to,
        if any.
    @type site: L{twisted.web.server.Site}

    @ivar factory: The L{twisted.web.http.HTTPFactory} object that constructed
        this stream's parent connection.
    @type factory: L{twisted.web.http.HTTPFactory}

    @ivar _producerProducing: Whether the producer stored in producer is
        currently producing data.
    @type _producerProducing: L{bool}

    @ivar _inboundDataBuffer: Any data that has been received from the network
        but has not yet been received by the consumer.
    @type _inboundDataBuffer: A L{collections.deque} containing L{bytes}

    @ivar _conn: A reference to the connection this stream belongs to.
    @type _conn: L{H2Connection}

    @ivar _request: A request object that this stream corresponds to.
    @type _request: L{twisted.web.iweb.IRequest}

    @ivar _buffer: A buffer containing data produced by the producer that could
        not be sent on the network at this time.
    @type _buffer: L{io.BytesIO}
    """
    transport = None

    def __init__(self, streamID, connection, headers, requestFactory, site, factory):
        """
        Initialize this HTTP/2 stream.

        @param streamID: The numerical stream ID that this object corresponds
            to.
        @type streamID: L{int}

        @param connection: The HTTP/2 connection this stream belongs to.
        @type connection: L{H2Connection}

        @param headers: The HTTP/2 request headers.
        @type headers: A L{list} of L{tuple}s of header name and header value,
            both as L{bytes}.

        @param requestFactory: A function that builds appropriate request
            request objects.
        @type requestFactory: A callable that returns a
            L{twisted.web.iweb.IRequest}.

        @param site: The L{twisted.web.server.Site} object this stream belongs
            to, if any.
        @type site: L{twisted.web.server.Site}

        @param factory: The L{twisted.web.http.HTTPFactory} object that
            constructed this stream's parent connection.
        @type factory: L{twisted.web.http.HTTPFactory}
        """
        self.streamID = streamID
        self.site = site
        self.factory = factory
        self.producing = True
        self.command = None
        self.path = None
        self.producer = None
        self._producerProducing = False
        self._hasStreamingProducer = None
        self._inboundDataBuffer = deque()
        self._conn = connection
        self._request = requestFactory(self, queued=False)
        self._buffer = io.BytesIO()
        self._convertHeaders(headers)

    def _convertHeaders(self, headers):
        """
        This method converts the HTTP/2 header set into something that looks
        like HTTP/1.1. In particular, it strips the 'special' headers and adds
        a Host: header.

        @param headers: The HTTP/2 header set.
        @type headers: A L{list} of L{tuple}s of header name and header value,
            both as L{bytes}.
        """
        gotLength = False
        for header in headers:
            if not header[0].startswith(b':'):
                gotLength = _addHeaderToRequest(self._request, header) or gotLength
            elif header[0] == b':method':
                self.command = header[1]
            elif header[0] == b':path':
                self.path = header[1]
            elif header[0] == b':authority':
                _addHeaderToRequest(self._request, (b'host', header[1]))
        if not gotLength:
            if self.command in (b'GET', b'HEAD'):
                self._request.gotLength(0)
            else:
                self._request.gotLength(None)
        self._request.parseCookies()
        expectContinue = self._request.requestHeaders.getRawHeaders(b'expect')
        if expectContinue and expectContinue[0].lower() == b'100-continue':
            self._send100Continue()

    def receiveDataChunk(self, data, flowControlledLength):
        """
        Called when the connection has received a chunk of data from the
        underlying transport. If the stream has been registered with a
        consumer, and is currently able to push data, immediately passes it
        through. Otherwise, buffers the chunk until we can start producing.

        @param data: The chunk of data that was received.
        @type data: L{bytes}

        @param flowControlledLength: The total flow controlled length of this
            chunk, which is used when we want to re-open the window. May be
            different to C{len(data)}.
        @type flowControlledLength: L{int}
        """
        if not self.producing:
            self._inboundDataBuffer.append((data, flowControlledLength))
        else:
            self._request.handleContentChunk(data)
            self._conn.openStreamWindow(self.streamID, flowControlledLength)

    def requestComplete(self):
        """
        Called by the L{H2Connection} when the all data for a request has been
        received. Currently, with the legacy L{twisted.web.http.Request}
        object, just calls requestReceived unless the producer wants us to be
        quiet.
        """
        if self.producing:
            self._request.requestReceived(self.command, self.path, b'HTTP/2')
        else:
            self._inboundDataBuffer.append((_END_STREAM_SENTINEL, None))

    def connectionLost(self, reason):
        """
        Called by the L{H2Connection} when a connection is lost or a stream is
        reset.

        @param reason: The reason the connection was lost.
        @type reason: L{str}
        """
        self._request.connectionLost(reason)

    def windowUpdated(self):
        """
        Called by the L{H2Connection} when this stream's flow control window
        has been opened.
        """
        if not self.producer:
            return
        if self._producerProducing:
            return
        remainingWindow = self._conn.remainingOutboundWindow(self.streamID)
        if not remainingWindow > 0:
            return
        self._producerProducing = True
        self.producer.resumeProducing()

    def flowControlBlocked(self):
        """
        Called by the L{H2Connection} when this stream's flow control window
        has been exhausted.
        """
        if not self.producer:
            return
        if self._producerProducing:
            self.producer.pauseProducing()
            self._producerProducing = False

    def writeHeaders(self, version, code, reason, headers):
        """
        Called by the consumer to write headers to the stream.

        @param version: The HTTP version.
        @type version: L{bytes}

        @param code: The status code.
        @type code: L{int}

        @param reason: The reason phrase. Ignored in HTTP/2.
        @type reason: L{bytes}

        @param headers: The HTTP response headers.
        @type headers: Any iterable of two-tuples of L{bytes}, representing header
            names and header values.
        """
        self._conn.writeHeaders(version, code, reason, headers, self.streamID)

    def requestDone(self, request):
        """
        Called by a consumer to clean up whatever permanent state is in use.

        @param request: The request calling the method.
        @type request: L{twisted.web.iweb.IRequest}
        """
        self._conn.endRequest(self.streamID)

    def _send100Continue(self):
        """
        Sends a 100 Continue response, used to signal to clients that further
        processing will be performed.
        """
        self._conn._send100Continue(self.streamID)

    def _respondToBadRequestAndDisconnect(self):
        """
        This is a quick and dirty way of responding to bad requests.

        As described by HTTP standard we should be patient and accept the
        whole request from the client before sending a polite bad request
        response, even in the case when clients send tons of data.

        Unlike in the HTTP/1.1 case, this does not actually disconnect the
        underlying transport: there's no need. This instead just sends a 400
        response and terminates the stream.
        """
        self._conn._respondToBadRequestAndDisconnect(self.streamID)

    def write(self, data):
        """
        Write a single chunk of data into a data frame.

        @param data: The data chunk to send.
        @type data: L{bytes}
        """
        self._conn.writeDataToStream(self.streamID, data)
        return

    def writeSequence(self, iovec):
        """
        Write a sequence of chunks of data into data frames.

        @param iovec: A sequence of chunks to send.
        @type iovec: An iterable of L{bytes} chunks.
        """
        for chunk in iovec:
            self.write(chunk)

    def loseConnection(self):
        """
        Close the connection after writing all pending data.
        """
        self._conn.endRequest(self.streamID)

    def abortConnection(self):
        """
        Forcefully abort the connection by sending a RstStream frame.
        """
        self._conn.abortRequest(self.streamID)

    def getPeer(self):
        """
        Get information about the peer.
        """
        return self._conn.getPeer()

    def getHost(self):
        """
        Similar to getPeer, but for this side of the connection.
        """
        return self._conn.getHost()

    def isSecure(self):
        """
        Returns L{True} if this channel is using a secure transport.

        @returns: L{True} if this channel is secure.
        @rtype: L{bool}
        """
        return self._conn._isSecure()

    def registerProducer(self, producer, streaming):
        """
        Register to receive data from a producer.

        This sets self to be a consumer for a producer.  When this object runs
        out of data (as when a send(2) call on a socket succeeds in moving the
        last data from a userspace buffer into a kernelspace buffer), it will
        ask the producer to resumeProducing().

        For L{IPullProducer} providers, C{resumeProducing} will be called once
        each time data is required.

        For L{IPushProducer} providers, C{pauseProducing} will be called
        whenever the write buffer fills up and C{resumeProducing} will only be
        called when it empties.

        @param producer: The producer to register.
        @type producer: L{IProducer} provider

        @param streaming: L{True} if C{producer} provides L{IPushProducer},
        L{False} if C{producer} provides L{IPullProducer}.
        @type streaming: L{bool}

        @raise RuntimeError: If a producer is already registered.

        @return: L{None}
        """
        if self.producer:
            raise ValueError('registering producer %s before previous one (%s) was unregistered' % (producer, self.producer))
        if not streaming:
            self.hasStreamingProducer = False
            producer = _PullToPush(producer, self)
            producer.startStreaming()
        else:
            self.hasStreamingProducer = True
        self.producer = producer
        self._producerProducing = True

    def unregisterProducer(self):
        """
        @see: L{IConsumer.unregisterProducer}
        """
        if self.producer is not None and (not self.hasStreamingProducer):
            self.producer.stopStreaming()
        self._producerProducing = False
        self.producer = None
        self.hasStreamingProducer = None

    def stopProducing(self):
        """
        @see: L{IProducer.stopProducing}
        """
        self.producing = False
        self.abortConnection()

    def pauseProducing(self):
        """
        @see: L{IPushProducer.pauseProducing}
        """
        self.producing = False

    def resumeProducing(self):
        """
        @see: L{IPushProducer.resumeProducing}
        """
        self.producing = True
        consumedLength = 0
        while self.producing and self._inboundDataBuffer:
            chunk, flowControlledLength = self._inboundDataBuffer.popleft()
            if chunk is _END_STREAM_SENTINEL:
                self.requestComplete()
            else:
                consumedLength += flowControlledLength
                self._request.handleContentChunk(chunk)
        self._conn.openStreamWindow(self.streamID, consumedLength)