import sys
import warnings
from functools import wraps
from io import BytesIO
from twisted.internet import defer, protocol
from twisted.python import failure
class _BackRelay(protocol.ProcessProtocol):
    """
    Trivial protocol for communicating with a process and turning its output
    into the result of a L{Deferred}.

    @ivar deferred: A L{Deferred} which will be called back with all of stdout
        and, if C{errortoo} is true, all of stderr as well (mixed together in
        one string).  If C{errortoo} is false and any bytes are received over
        stderr, this will fire with an L{_UnexpectedErrorOutput} instance and
        the attribute will be set to L{None}.

    @ivar onProcessEnded: If C{errortoo} is false and bytes are received over
        stderr, this attribute will refer to a L{Deferred} which will be called
        back when the process ends.  This C{Deferred} is also associated with
        the L{_UnexpectedErrorOutput} which C{deferred} fires with earlier in
        this case so that users can determine when the process has actually
        ended, in addition to knowing when bytes have been received via stderr.
    """

    def __init__(self, deferred, errortoo=0):
        self.deferred = deferred
        self.s = BytesIO()
        if errortoo:
            self.errReceived = self.errReceivedIsGood
        else:
            self.errReceived = self.errReceivedIsBad

    def errReceivedIsBad(self, text):
        if self.deferred is not None:
            self.onProcessEnded = defer.Deferred()
            err = _UnexpectedErrorOutput(text, self.onProcessEnded)
            self.deferred.errback(failure.Failure(err))
            self.deferred = None
            self.transport.loseConnection()

    def errReceivedIsGood(self, text):
        self.s.write(text)

    def outReceived(self, text):
        self.s.write(text)

    def processEnded(self, reason):
        if self.deferred is not None:
            self.deferred.callback(self.s.getvalue())
        elif self.onProcessEnded is not None:
            self.onProcessEnded.errback(reason)