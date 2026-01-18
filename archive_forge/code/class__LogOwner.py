from socket import AF_INET, AF_INET6, inet_pton
from typing import Iterable, List, Optional
from zope.interface import implementer
from twisted.internet import interfaces, main
from twisted.python import failure, reflect
from twisted.python.compat import lazyByteSlice
@implementer(interfaces.ILoggingContext)
class _LogOwner:
    """
    Mixin to help implement L{interfaces.ILoggingContext} for transports which
    have a protocol, the log prefix of which should also appear in the
    transport's log prefix.
    """

    def _getLogPrefix(self, applicationObject: object) -> str:
        """
        Determine the log prefix to use for messages related to
        C{applicationObject}, which may or may not be an
        L{interfaces.ILoggingContext} provider.

        @return: A C{str} giving the log prefix to use.
        """
        if interfaces.ILoggingContext.providedBy(applicationObject):
            return applicationObject.logPrefix()
        return applicationObject.__class__.__name__

    def logPrefix(self):
        """
        Override this method to insert custom logging behavior.  Its
        return value will be inserted in front of every line.  It may
        be called more times than the number of output lines.
        """
        return '-'