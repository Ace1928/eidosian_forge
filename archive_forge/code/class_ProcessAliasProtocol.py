import os
import tempfile
from zope.interface import implementer
from twisted.internet import defer, protocol, reactor
from twisted.mail import smtp
from twisted.mail.interfaces import IAlias
from twisted.python import failure, log
class ProcessAliasProtocol(protocol.ProcessProtocol):
    """
    A process protocol which errbacks a deferred when the associated
    process ends.

    @type onEnd: L{None} or L{Deferred <defer.Deferred>}
    @ivar onEnd: If set, a deferred on which to errback when the process ends.
    """
    onEnd = None

    def processEnded(self, reason):
        """
        Call an errback.

        @type reason: L{Failure <failure.Failure>}
        @param reason: The reason the child process terminated.
        """
        if self.onEnd is not None:
            self.onEnd.errback(reason)