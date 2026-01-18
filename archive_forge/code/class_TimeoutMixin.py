import sys
from typing import Optional, Type
from zope.interface import directlyProvides, providedBy
from twisted.internet import error, interfaces
from twisted.internet.interfaces import ILoggingContext
from twisted.internet.protocol import ClientFactory, Protocol, ServerFactory
from twisted.python import log
class TimeoutMixin:
    """
    Mixin for protocols which wish to timeout connections.

    Protocols that mix this in have a single timeout, set using L{setTimeout}.
    When the timeout is hit, L{timeoutConnection} is called, which, by
    default, closes the connection.

    @cvar timeOut: The number of seconds after which to timeout the connection.
    """
    timeOut: Optional[int] = None
    __timeoutCall = None

    def callLater(self, period, func):
        """
        Wrapper around
        L{reactor.callLater<twisted.internet.interfaces.IReactorTime.callLater>}
        for test purpose.
        """
        from twisted.internet import reactor
        return reactor.callLater(period, func)

    def resetTimeout(self):
        """
        Reset the timeout count down.

        If the connection has already timed out, then do nothing.  If the
        timeout has been cancelled (probably using C{setTimeout(None)}), also
        do nothing.

        It's often a good idea to call this when the protocol has received
        some meaningful input from the other end of the connection.  "I've got
        some data, they're still there, reset the timeout".
        """
        if self.__timeoutCall is not None and self.timeOut is not None:
            self.__timeoutCall.reset(self.timeOut)

    def setTimeout(self, period):
        """
        Change the timeout period

        @type period: C{int} or L{None}
        @param period: The period, in seconds, to change the timeout to, or
        L{None} to disable the timeout.
        """
        prev = self.timeOut
        self.timeOut = period
        if self.__timeoutCall is not None:
            if period is None:
                try:
                    self.__timeoutCall.cancel()
                except (error.AlreadyCancelled, error.AlreadyCalled):
                    pass
                self.__timeoutCall = None
            else:
                self.__timeoutCall.reset(period)
        elif period is not None:
            self.__timeoutCall = self.callLater(period, self.__timedOut)
        return prev

    def __timedOut(self):
        self.__timeoutCall = None
        self.timeoutConnection()

    def timeoutConnection(self):
        """
        Called when the connection times out.

        Override to define behavior other than dropping the connection.
        """
        self.transport.loseConnection()