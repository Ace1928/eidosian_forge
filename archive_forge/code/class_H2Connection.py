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
@implementer(IProtocol, IPushProducer)
class H2Connection(Protocol, TimeoutMixin):
    """
    A class representing a single HTTP/2 connection.

    This implementation of L{IProtocol} works hand in hand with L{H2Stream}.
    This is because we have the requirement to register multiple producers for
    a single HTTP/2 connection, one for each stream. The standard Twisted
    interfaces don't really allow for this, so instead there's a custom
    interface between the two objects that allows them to work hand-in-hand here.

    @ivar conn: The HTTP/2 connection state machine.
    @type conn: L{h2.connection.H2Connection}

    @ivar streams: A mapping of stream IDs to L{H2Stream} objects, used to call
        specific methods on streams when events occur.
    @type streams: L{dict}, mapping L{int} stream IDs to L{H2Stream} objects.

    @ivar priority: A HTTP/2 priority tree used to ensure that responses are
        prioritised appropriately.
    @type priority: L{priority.PriorityTree}

    @ivar _consumerBlocked: A flag tracking whether or not the L{IConsumer}
        that is consuming this data has asked us to stop producing.
    @type _consumerBlocked: L{bool}

    @ivar _sendingDeferred: A L{Deferred} used to restart the data-sending loop
        when more response data has been produced. Will not be present if there
        is outstanding data still to send.
    @type _consumerBlocked: A L{twisted.internet.defer.Deferred}, or L{None}

    @ivar _outboundStreamQueues: A map of stream IDs to queues, used to store
        data blocks that are yet to be sent on the connection. These are used
        both to handle producers that do not respect L{IConsumer} but also to
        allow priority to multiplex data appropriately.
    @type _outboundStreamQueues: A L{dict} mapping L{int} stream IDs to
        L{collections.deque} queues, which contain either L{bytes} objects or
        C{_END_STREAM_SENTINEL}.

    @ivar _sender: A handle to the data-sending loop, allowing it to be
        terminated if needed.
    @type _sender: L{twisted.internet.task.LoopingCall}

    @ivar abortTimeout: The number of seconds to wait after we attempt to shut
        the transport down cleanly to give up and forcibly terminate it. This
        is only used when we time a connection out, to prevent errors causing
        the FD to get leaked. If this is L{None}, we will wait forever.
    @type abortTimeout: L{int}

    @ivar _abortingCall: The L{twisted.internet.base.DelayedCall} that will be
        used to forcibly close the transport if it doesn't close cleanly.
    @type _abortingCall: L{twisted.internet.base.DelayedCall}
    """
    factory = None
    site = None
    abortTimeout = 15
    _log = Logger()
    _abortingCall = None

    def __init__(self, reactor=None):
        config = h2.config.H2Configuration(client_side=False, header_encoding=None)
        self.conn = h2.connection.H2Connection(config=config)
        self.streams = {}
        self.priority = priority.PriorityTree()
        self._consumerBlocked = None
        self._sendingDeferred = None
        self._outboundStreamQueues = {}
        self._streamCleanupCallbacks = {}
        self._stillProducing = True
        self._maxBufferedControlFrameBytes = 1024 * 17
        self._bufferedControlFrames = deque()
        self._bufferedControlFrameBytes = 0
        if reactor is None:
            from twisted.internet import reactor
        self._reactor = reactor
        self._reactor.callLater(0, self._sendPrioritisedData)

    def connectionMade(self):
        """
        Called by the reactor when a connection is received. May also be called
        by the L{twisted.web.http._GenericHTTPChannelProtocol} during upgrade
        to HTTP/2.
        """
        self.setTimeout(self.timeOut)
        self.conn.initiate_connection()
        self.transport.write(self.conn.data_to_send())

    def dataReceived(self, data):
        """
        Called whenever a chunk of data is received from the transport.

        @param data: The data received from the transport.
        @type data: L{bytes}
        """
        try:
            events = self.conn.receive_data(data)
        except h2.exceptions.ProtocolError:
            stillActive = self._tryToWriteControlData()
            if stillActive:
                self.transport.loseConnection()
                self.connectionLost(Failure(), _cancelTimeouts=False)
            return
        self.resetTimeout()
        for event in events:
            if isinstance(event, h2.events.RequestReceived):
                self._requestReceived(event)
            elif isinstance(event, h2.events.DataReceived):
                self._requestDataReceived(event)
            elif isinstance(event, h2.events.StreamEnded):
                self._requestEnded(event)
            elif isinstance(event, h2.events.StreamReset):
                self._requestAborted(event)
            elif isinstance(event, h2.events.WindowUpdated):
                self._handleWindowUpdate(event)
            elif isinstance(event, h2.events.PriorityUpdated):
                self._handlePriorityUpdate(event)
            elif isinstance(event, h2.events.ConnectionTerminated):
                self.transport.loseConnection()
                self.connectionLost(Failure(ConnectionLost('Remote peer sent GOAWAY')), _cancelTimeouts=False)
        self._tryToWriteControlData()

    def timeoutConnection(self):
        """
        Called when the connection has been inactive for
        L{self.timeOut<twisted.protocols.policies.TimeoutMixin.timeOut>}
        seconds. Cleanly tears the connection down, attempting to notify the
        peer if needed.

        We override this method to add two extra bits of functionality:

         - We want to log the timeout.
         - We want to send a GOAWAY frame indicating that the connection is
           being terminated, and whether it was clean or not. We have to do this
           before the connection is torn down.
        """
        self._log.info('Timing out client {client}', client=self.transport.getPeer())
        if self.conn.open_outbound_streams > 0 or self.conn.open_inbound_streams > 0:
            error_code = h2.errors.ErrorCodes.PROTOCOL_ERROR
        else:
            error_code = h2.errors.ErrorCodes.NO_ERROR
        self.conn.close_connection(error_code=error_code)
        self.transport.write(self.conn.data_to_send())
        if self.abortTimeout is not None:
            self._abortingCall = self.callLater(self.abortTimeout, self.forceAbortClient)
        self.transport.loseConnection()

    def forceAbortClient(self):
        """
        Called if C{abortTimeout} seconds have passed since the timeout fired,
        and the connection still hasn't gone away. This can really only happen
        on extremely bad connections or when clients are maliciously attempting
        to keep connections open.
        """
        self._log.info('Forcibly timing out client: {client}', client=self.transport.getPeer())
        self._abortingCall = None
        self.transport.abortConnection()

    def connectionLost(self, reason, _cancelTimeouts=True):
        """
        Called when the transport connection is lost.

        Informs all outstanding response handlers that the connection
        has been lost, and cleans up all internal state.

        @param reason: See L{IProtocol.connectionLost}

        @param _cancelTimeouts: Propagate the C{reason} to this
            connection's streams but don't cancel any timers, so that
            peers who never read the data we've written are eventually
            timed out.
        """
        self._stillProducing = False
        if _cancelTimeouts:
            self.setTimeout(None)
        for stream in self.streams.values():
            stream.connectionLost(reason)
        for streamID in list(self.streams.keys()):
            self._requestDone(streamID)
        if _cancelTimeouts and self._abortingCall is not None:
            self._abortingCall.cancel()
            self._abortingCall = None

    def stopProducing(self):
        """
        Stop producing data.

        This tells the L{H2Connection} that its consumer has died, so it must
        stop producing data for good.
        """
        self.connectionLost(Failure(ConnectionLost('Producing stopped')))

    def pauseProducing(self):
        """
        Pause producing data.

        Tells the L{H2Connection} that it has produced too much data to process
        for the time being, and to stop until resumeProducing() is called.
        """
        self._consumerBlocked = Deferred()
        self._consumerBlocked.addCallback(self._flushBufferedControlData)

    def resumeProducing(self):
        """
        Resume producing data.

        This tells the L{H2Connection} to re-add itself to the main loop and
        produce more data for the consumer.
        """
        if self._consumerBlocked is not None:
            d = self._consumerBlocked
            self._consumerBlocked = None
            d.callback(None)

    def _sendPrioritisedData(self, *args):
        """
        The data sending loop. This function repeatedly calls itself, either
        from L{Deferred}s or from
        L{reactor.callLater<twisted.internet.interfaces.IReactorTime.callLater>}

        This function sends data on streams according to the rules of HTTP/2
        priority. It ensures that the data from each stream is interleved
        according to the priority signalled by the client, making sure that the
        connection is used with maximal efficiency.

        This function will execute if data is available: if all data is
        exhausted, the function will place a deferred onto the L{H2Connection}
        object and wait until it is called to resume executing.
        """
        if not self._stillProducing:
            return
        stream = None
        while stream is None:
            try:
                stream = next(self.priority)
            except priority.DeadlockError:
                assert self._sendingDeferred is None
                self._sendingDeferred = Deferred()
                self._sendingDeferred.addCallback(self._sendPrioritisedData)
                return
        if self._consumerBlocked is not None:
            self._consumerBlocked.addCallback(self._sendPrioritisedData)
            return
        self.resetTimeout()
        remainingWindow = self.conn.local_flow_control_window(stream)
        frameData = self._outboundStreamQueues[stream].popleft()
        maxFrameSize = min(self.conn.max_outbound_frame_size, remainingWindow)
        if frameData is _END_STREAM_SENTINEL:
            self.conn.end_stream(stream)
            self.transport.write(self.conn.data_to_send())
            self._requestDone(stream)
        else:
            if len(frameData) > maxFrameSize:
                excessData = frameData[maxFrameSize:]
                frameData = frameData[:maxFrameSize]
                self._outboundStreamQueues[stream].appendleft(excessData)
            if frameData:
                self.conn.send_data(stream, frameData)
                self.transport.write(self.conn.data_to_send())
            if not self._outboundStreamQueues[stream]:
                self.priority.block(stream)
            if self.remainingOutboundWindow(stream) <= 0:
                self.streams[stream].flowControlBlocked()
        self._reactor.callLater(0, self._sendPrioritisedData)

    def _requestReceived(self, event):
        """
        Internal handler for when a request has been received.

        @param event: The Hyper-h2 event that encodes information about the
            received request.
        @type event: L{h2.events.RequestReceived}
        """
        stream = H2Stream(event.stream_id, self, event.headers, self.requestFactory, self.site, self.factory)
        self.streams[event.stream_id] = stream
        self._streamCleanupCallbacks[event.stream_id] = Deferred()
        self._outboundStreamQueues[event.stream_id] = deque()
        try:
            self.priority.insert_stream(event.stream_id)
        except priority.DuplicateStreamError:
            pass
        else:
            self.priority.block(event.stream_id)

    def _requestDataReceived(self, event):
        """
        Internal handler for when a chunk of data is received for a given
        request.

        @param event: The Hyper-h2 event that encodes information about the
            received data.
        @type event: L{h2.events.DataReceived}
        """
        stream = self.streams[event.stream_id]
        stream.receiveDataChunk(event.data, event.flow_controlled_length)

    def _requestEnded(self, event):
        """
        Internal handler for when a request is complete, and we expect no
        further data for that request.

        @param event: The Hyper-h2 event that encodes information about the
            completed stream.
        @type event: L{h2.events.StreamEnded}
        """
        stream = self.streams[event.stream_id]
        stream.requestComplete()

    def _requestAborted(self, event):
        """
        Internal handler for when a request is aborted by a remote peer.

        @param event: The Hyper-h2 event that encodes information about the
            reset stream.
        @type event: L{h2.events.StreamReset}
        """
        stream = self.streams[event.stream_id]
        stream.connectionLost(Failure(ConnectionLost('Stream reset with code %s' % event.error_code)))
        self._requestDone(event.stream_id)

    def _handlePriorityUpdate(self, event):
        """
        Internal handler for when a stream priority is updated.

        @param event: The Hyper-h2 event that encodes information about the
            stream reprioritization.
        @type event: L{h2.events.PriorityUpdated}
        """
        try:
            self.priority.reprioritize(stream_id=event.stream_id, depends_on=event.depends_on or None, weight=event.weight, exclusive=event.exclusive)
        except priority.MissingStreamError:
            self.priority.insert_stream(stream_id=event.stream_id, depends_on=event.depends_on or None, weight=event.weight, exclusive=event.exclusive)
            self.priority.block(event.stream_id)

    def writeHeaders(self, version, code, reason, headers, streamID):
        """
        Called by L{twisted.web.http.Request} objects to write a complete set
        of HTTP headers to a stream.

        @param version: The HTTP version in use. Unused in HTTP/2.
        @type version: L{bytes}

        @param code: The HTTP status code to write.
        @type code: L{bytes}

        @param reason: The HTTP reason phrase to write. Unused in HTTP/2.
        @type reason: L{bytes}

        @param headers: The headers to write to the stream.
        @type headers: L{twisted.web.http_headers.Headers}

        @param streamID: The ID of the stream to write the headers to.
        @type streamID: L{int}
        """
        headers.insert(0, (b':status', code))
        try:
            self.conn.send_headers(streamID, headers)
        except h2.exceptions.StreamClosedError:
            return
        else:
            self._tryToWriteControlData()

    def writeDataToStream(self, streamID, data):
        """
        May be called by L{H2Stream} objects to write response data to a given
        stream. Writes a single data frame.

        @param streamID: The ID of the stream to write the data to.
        @type streamID: L{int}

        @param data: The data chunk to write to the stream.
        @type data: L{bytes}
        """
        self._outboundStreamQueues[streamID].append(data)
        if self.conn.local_flow_control_window(streamID) > 0:
            self.priority.unblock(streamID)
            if self._sendingDeferred is not None:
                d = self._sendingDeferred
                self._sendingDeferred = None
                d.callback(streamID)
        if self.remainingOutboundWindow(streamID) <= 0:
            self.streams[streamID].flowControlBlocked()

    def endRequest(self, streamID):
        """
        Called by L{H2Stream} objects to signal completion of a response.

        @param streamID: The ID of the stream to write the data to.
        @type streamID: L{int}
        """
        self._outboundStreamQueues[streamID].append(_END_STREAM_SENTINEL)
        self.priority.unblock(streamID)
        if self._sendingDeferred is not None:
            d = self._sendingDeferred
            self._sendingDeferred = None
            d.callback(streamID)

    def abortRequest(self, streamID):
        """
        Called by L{H2Stream} objects to request early termination of a stream.
        This emits a RstStream frame and then removes all stream state.

        @param streamID: The ID of the stream to write the data to.
        @type streamID: L{int}
        """
        self.conn.reset_stream(streamID)
        stillActive = self._tryToWriteControlData()
        if stillActive:
            self._requestDone(streamID)

    def _requestDone(self, streamID):
        """
        Called internally by the data sending loop to clean up state that was
        being used for the stream. Called when the stream is complete.

        @param streamID: The ID of the stream to clean up state for.
        @type streamID: L{int}
        """
        del self._outboundStreamQueues[streamID]
        self.priority.remove_stream(streamID)
        del self.streams[streamID]
        cleanupCallback = self._streamCleanupCallbacks.pop(streamID)
        cleanupCallback.callback(streamID)

    def remainingOutboundWindow(self, streamID):
        """
        Called to determine how much room is left in the send window for a
        given stream. Allows us to handle blocking and unblocking producers.

        @param streamID: The ID of the stream whose flow control window we'll
            check.
        @type streamID: L{int}

        @return: The amount of room remaining in the send window for the given
            stream, including the data queued to be sent.
        @rtype: L{int}
        """
        windowSize = self.conn.local_flow_control_window(streamID)
        sendQueue = self._outboundStreamQueues[streamID]
        alreadyConsumed = sum((len(chunk) for chunk in sendQueue if chunk is not _END_STREAM_SENTINEL))
        return windowSize - alreadyConsumed

    def _handleWindowUpdate(self, event):
        """
        Manage flow control windows.

        Streams that are blocked on flow control will register themselves with
        the connection. This will fire deferreds that wake those streams up and
        allow them to continue processing.

        @param event: The Hyper-h2 event that encodes information about the
            flow control window change.
        @type event: L{h2.events.WindowUpdated}
        """
        streamID = event.stream_id
        if streamID:
            if not self._streamIsActive(streamID):
                return
            if self._outboundStreamQueues.get(streamID):
                self.priority.unblock(streamID)
            self.streams[streamID].windowUpdated()
        else:
            for stream in self.streams.values():
                stream.windowUpdated()
                if self._outboundStreamQueues.get(stream.streamID):
                    self.priority.unblock(stream.streamID)

    def getPeer(self):
        """
        Get the remote address of this connection.

        Treat this method with caution.  It is the unfortunate result of the
        CGI and Jabber standards, but should not be considered reliable for
        the usual host of reasons; port forwarding, proxying, firewalls, IP
        masquerading, etc.

        @return: An L{IAddress} provider.
        """
        return self.transport.getPeer()

    def getHost(self):
        """
        Similar to getPeer, but returns an address describing this side of the
        connection.

        @return: An L{IAddress} provider.
        """
        return self.transport.getHost()

    def openStreamWindow(self, streamID, increment):
        """
        Open the stream window by a given increment.

        @param streamID: The ID of the stream whose window needs to be opened.
        @type streamID: L{int}

        @param increment: The amount by which the stream window must be
        incremented.
        @type increment: L{int}
        """
        self.conn.acknowledge_received_data(increment, streamID)
        self._tryToWriteControlData()

    def _isSecure(self):
        """
        Returns L{True} if this channel is using a secure transport.

        @returns: L{True} if this channel is secure.
        @rtype: L{bool}
        """
        return ISSLTransport(self.transport, None) is not None

    def _send100Continue(self, streamID):
        """
        Sends a 100 Continue response, used to signal to clients that further
        processing will be performed.

        @param streamID: The ID of the stream that needs the 100 Continue
        response
        @type streamID: L{int}
        """
        headers = [(b':status', b'100')]
        self.conn.send_headers(headers=headers, stream_id=streamID)
        self._tryToWriteControlData()

    def _respondToBadRequestAndDisconnect(self, streamID):
        """
        This is a quick and dirty way of responding to bad requests.

        As described by HTTP standard we should be patient and accept the
        whole request from the client before sending a polite bad request
        response, even in the case when clients send tons of data.

        Unlike in the HTTP/1.1 case, this does not actually disconnect the
        underlying transport: there's no need. This instead just sends a 400
        response and terminates the stream.

        @param streamID: The ID of the stream that needs the 100 Continue
        response
        @type streamID: L{int}
        """
        headers = [(b':status', b'400')]
        self.conn.send_headers(headers=headers, stream_id=streamID, end_stream=True)
        stillActive = self._tryToWriteControlData()
        if stillActive:
            stream = self.streams[streamID]
            stream.connectionLost(Failure(ConnectionLost('Invalid request')))
            self._requestDone(streamID)

    def _streamIsActive(self, streamID):
        """
        Checks whether Twisted has still got state for a given stream and so
        can process events for that stream.

        @param streamID: The ID of the stream that needs processing.
        @type streamID: L{int}

        @return: Whether the stream still has state allocated.
        @rtype: L{bool}
        """
        return streamID in self.streams

    def _tryToWriteControlData(self):
        """
        Checks whether the connection is blocked on flow control and,
        if it isn't, writes any buffered control data.

        @return: L{True} if the connection is still active and
            L{False} if it was aborted because too many bytes have
            been written but not consumed by the other end.
        """
        bufferedBytes = self.conn.data_to_send()
        if not bufferedBytes:
            return True
        if self._consumerBlocked is None and (not self._bufferedControlFrames):
            self.transport.write(bufferedBytes)
            return True
        else:
            self._bufferedControlFrames.append(bufferedBytes)
            self._bufferedControlFrameBytes += len(bufferedBytes)
            if self._bufferedControlFrameBytes >= self._maxBufferedControlFrameBytes:
                maxBuffCtrlFrameBytes = self._maxBufferedControlFrameBytes
                self._log.error('Maximum number of control frame bytes buffered: {bufferedControlFrameBytes} > = {maxBufferedControlFrameBytes}. Aborting connection to client: {client} ', bufferedControlFrameBytes=self._bufferedControlFrameBytes, maxBufferedControlFrameBytes=maxBuffCtrlFrameBytes, client=self.transport.getPeer())
                self.transport.abortConnection()
                self.connectionLost(Failure(ExcessiveBufferingError()))
                return False
            return True

    def _flushBufferedControlData(self, *args):
        """
        Called when the connection is marked writable again after being marked unwritable.
        Attempts to flush buffered control data if there is any.
        """
        while self._consumerBlocked is None and self._bufferedControlFrames:
            nextWrite = self._bufferedControlFrames.popleft()
            self._bufferedControlFrameBytes -= len(nextWrite)
            self.transport.write(nextWrite)