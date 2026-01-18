import re
from zope.interface import implementer
from twisted.internet.defer import (
from twisted.internet.error import ConnectionDone
from twisted.internet.interfaces import IConsumer, IPushProducer
from twisted.internet.protocol import Protocol
from twisted.logger import Logger
from twisted.protocols.basic import LineReceiver
from twisted.python.compat import networkString
from twisted.python.components import proxyForInterface
from twisted.python.failure import Failure
from twisted.python.reflect import fullyQualifiedName
from twisted.web.http import (
from twisted.web.http_headers import Headers
from twisted.web.iweb import UNKNOWN_LENGTH, IClientRequest, IResponse
class HTTP11ClientProtocol(Protocol):
    """
    L{HTTP11ClientProtocol} is an implementation of the HTTP 1.1 client
    protocol.  It supports as few features as possible.

    @ivar _parser: After a request is issued, the L{HTTPClientParser} to
        which received data making up the response to that request is
        delivered.

    @ivar _finishedRequest: After a request is issued, the L{Deferred} which
        will fire when a L{Response} object corresponding to that request is
        available.  This allows L{HTTP11ClientProtocol} to fail the request
        if there is a connection or parsing problem.

    @ivar _currentRequest: After a request is issued, the L{Request}
        instance used to make that request.  This allows
        L{HTTP11ClientProtocol} to stop request generation if necessary (for
        example, if the connection is lost).

    @ivar _transportProxy: After a request is issued, the
        L{TransportProxyProducer} to which C{_parser} is connected.  This
        allows C{_parser} to pause and resume the transport in a way which
        L{HTTP11ClientProtocol} can exert some control over.

    @ivar _responseDeferred: After a request is issued, the L{Deferred} from
        C{_parser} which will fire with a L{Response} when one has been
        received.  This is eventually chained with C{_finishedRequest}, but
        only in certain cases to avoid double firing that Deferred.

    @ivar _state: Indicates what state this L{HTTP11ClientProtocol} instance
        is in with respect to transmission of a request and reception of a
        response.  This may be one of the following strings:

          - QUIESCENT: This is the state L{HTTP11ClientProtocol} instances
            start in.  Nothing is happening: no request is being sent and no
            response is being received or expected.

          - TRANSMITTING: When a request is made (via L{request}), the
            instance moves to this state.  L{Request.writeTo} has been used
            to start to send a request but it has not yet finished.

          - TRANSMITTING_AFTER_RECEIVING_RESPONSE: The server has returned a
            complete response but the request has not yet been fully sent
            yet.  The instance will remain in this state until the request
            is fully sent.

          - GENERATION_FAILED: There was an error while the request.  The
            request was not fully sent to the network.

          - WAITING: The request was fully sent to the network.  The
            instance is now waiting for the response to be fully received.

          - ABORTING: Application code has requested that the HTTP connection
            be aborted.

          - CONNECTION_LOST: The connection has been lost.
    @type _state: C{str}

    @ivar _abortDeferreds: A list of C{Deferred} instances that will fire when
        the connection is lost.
    """
    _state = 'QUIESCENT'
    _parser = None
    _finishedRequest = None
    _currentRequest = None
    _transportProxy = None
    _responseDeferred = None
    _log = Logger()

    def __init__(self, quiescentCallback=lambda c: None):
        self._quiescentCallback = quiescentCallback
        self._abortDeferreds = []

    @property
    def state(self):
        return self._state

    def request(self, request):
        """
        Issue C{request} over C{self.transport} and return a L{Deferred} which
        will fire with a L{Response} instance or an error.

        @param request: The object defining the parameters of the request to
           issue.
        @type request: L{Request}

        @rtype: L{Deferred}
        @return: The deferred may errback with L{RequestGenerationFailed} if
            the request was not fully written to the transport due to a local
            error.  It may errback with L{RequestTransmissionFailed} if it was
            not fully written to the transport due to a network error.  It may
            errback with L{ResponseFailed} if the request was sent (not
            necessarily received) but some or all of the response was lost.  It
            may errback with L{RequestNotSent} if it is not possible to send
            any more requests using this L{HTTP11ClientProtocol}.
        """
        if self._state != 'QUIESCENT':
            return fail(RequestNotSent())
        self._state = 'TRANSMITTING'
        _requestDeferred = maybeDeferred(request.writeTo, self.transport)

        def cancelRequest(ign):
            if self._state in ('TRANSMITTING', 'TRANSMITTING_AFTER_RECEIVING_RESPONSE'):
                _requestDeferred.cancel()
            else:
                self.transport.abortConnection()
                self._disconnectParser(Failure(CancelledError()))
        self._finishedRequest = Deferred(cancelRequest)
        self._currentRequest = request
        self._transportProxy = TransportProxyProducer(self.transport)
        self._parser = HTTPClientParser(request, self._finishResponse)
        self._parser.makeConnection(self._transportProxy)
        self._responseDeferred = self._parser._responseDeferred

        def cbRequestWritten(ignored):
            if self._state == 'TRANSMITTING':
                self._state = 'WAITING'
                self._responseDeferred.chainDeferred(self._finishedRequest)

        def ebRequestWriting(err):
            if self._state == 'TRANSMITTING':
                self._state = 'GENERATION_FAILED'
                self.transport.abortConnection()
                self._finishedRequest.errback(Failure(RequestGenerationFailed([err])))
            else:
                self._log.failure('Error writing request, but not in valid state to finalize request: {state}', failure=err, state=self._state)
        _requestDeferred.addCallbacks(cbRequestWritten, ebRequestWriting)
        return self._finishedRequest

    def _finishResponse(self, rest):
        """
        Called by an L{HTTPClientParser} to indicate that it has parsed a
        complete response.

        @param rest: A C{bytes} giving any trailing bytes which were given to
            the L{HTTPClientParser} which were not part of the response it
            was parsing.
        """
    _finishResponse = makeStatefulDispatcher('finishResponse', _finishResponse)

    def _finishResponse_WAITING(self, rest):
        if self._state == 'WAITING':
            self._state = 'QUIESCENT'
        else:
            self._state = 'TRANSMITTING_AFTER_RECEIVING_RESPONSE'
            self._responseDeferred.chainDeferred(self._finishedRequest)
        if self._parser is None:
            return
        reason = ConnectionDone('synthetic!')
        connHeaders = self._parser.connHeaders.getRawHeaders(b'connection', ())
        if b'close' in connHeaders or self._state != 'QUIESCENT' or (not self._currentRequest.persistent):
            self._giveUp(Failure(reason))
        else:
            self.transport.resumeProducing()
            try:
                self._quiescentCallback(self)
            except BaseException:
                self._log.failure('')
                self.transport.loseConnection()
            self._disconnectParser(reason)
    _finishResponse_TRANSMITTING = _finishResponse_WAITING

    def _disconnectParser(self, reason):
        """
        If there is still a parser, call its C{connectionLost} method with the
        given reason.  If there is not, do nothing.

        @type reason: L{Failure}
        """
        if self._parser is not None:
            parser = self._parser
            self._parser = None
            self._currentRequest = None
            self._finishedRequest = None
            self._responseDeferred = None
            self._transportProxy.stopProxying()
            self._transportProxy = None
            parser.connectionLost(reason)

    def _giveUp(self, reason):
        """
        Lose the underlying connection and disconnect the parser with the given
        L{Failure}.

        Use this method instead of calling the transport's loseConnection
        method directly otherwise random things will break.
        """
        self.transport.loseConnection()
        self._disconnectParser(reason)

    def dataReceived(self, bytes):
        """
        Handle some stuff from some place.
        """
        try:
            self._parser.dataReceived(bytes)
        except BaseException:
            self._giveUp(Failure())

    def connectionLost(self, reason):
        """
        The underlying transport went away.  If appropriate, notify the parser
        object.
        """
    connectionLost = makeStatefulDispatcher('connectionLost', connectionLost)

    def _connectionLost_QUIESCENT(self, reason):
        """
        Nothing is currently happening.  Move to the C{'CONNECTION_LOST'}
        state but otherwise do nothing.
        """
        self._state = 'CONNECTION_LOST'

    def _connectionLost_GENERATION_FAILED(self, reason):
        """
        The connection was in an inconsistent state.  Move to the
        C{'CONNECTION_LOST'} state but otherwise do nothing.
        """
        self._state = 'CONNECTION_LOST'

    def _connectionLost_TRANSMITTING(self, reason):
        """
        Fail the L{Deferred} for the current request, notify the request
        object that it does not need to continue transmitting itself, and
        move to the C{'CONNECTION_LOST'} state.
        """
        self._state = 'CONNECTION_LOST'
        self._finishedRequest.errback(Failure(RequestTransmissionFailed([reason])))
        del self._finishedRequest
        self._currentRequest.stopWriting()

    def _connectionLost_TRANSMITTING_AFTER_RECEIVING_RESPONSE(self, reason):
        """
        Move to the C{'CONNECTION_LOST'} state.
        """
        self._state = 'CONNECTION_LOST'

    def _connectionLost_WAITING(self, reason):
        """
        Disconnect the response parser so that it can propagate the event as
        necessary (for example, to call an application protocol's
        C{connectionLost} method, or to fail a request L{Deferred}) and move
        to the C{'CONNECTION_LOST'} state.
        """
        self._disconnectParser(reason)
        self._state = 'CONNECTION_LOST'

    def _connectionLost_ABORTING(self, reason):
        """
        Disconnect the response parser with a L{ConnectionAborted} failure, and
        move to the C{'CONNECTION_LOST'} state.
        """
        self._disconnectParser(Failure(ConnectionAborted()))
        self._state = 'CONNECTION_LOST'
        for d in self._abortDeferreds:
            d.callback(None)
        self._abortDeferreds = []

    def abort(self):
        """
        Close the connection and cause all outstanding L{request} L{Deferred}s
        to fire with an error.
        """
        if self._state == 'CONNECTION_LOST':
            return succeed(None)
        self.transport.loseConnection()
        self._state = 'ABORTING'
        d = Deferred()
        self._abortDeferreds.append(d)
        return d