from io import StringIO
from twisted.internet import defer
from twisted.python import log
from twisted.python.reflect import qual
from twisted.spread import flavors, jelly, pb
from twisted.test.iosim import connectedServerAndClient
from twisted.trial import unittest
class SimpleRoot(pb.Root):

    def remote_asynchronousException(self):
        """
        Fail asynchronously with a non-pb.Error exception.
        """
        return defer.fail(AsynchronousException('remote asynchronous exception'))

    def remote_synchronousException(self):
        """
        Fail synchronously with a non-pb.Error exception.
        """
        raise SynchronousException('remote synchronous exception')

    def remote_asynchronousError(self):
        """
        Fail asynchronously with a pb.Error exception.
        """
        return defer.fail(AsynchronousError('remote asynchronous error'))

    def remote_synchronousError(self):
        """
        Fail synchronously with a pb.Error exception.
        """
        raise SynchronousError('remote synchronous error')

    def remote_unknownError(self):
        """
        Fail with error that is not known to client.
        """

        class UnknownError(pb.Error):
            pass
        raise UnknownError("I'm not known to client!")

    def remote_jelly(self):
        self.raiseJelly()

    def remote_security(self):
        self.raiseSecurity()

    def remote_deferredJelly(self):
        d = defer.Deferred()
        d.addCallback(self.raiseJelly)
        d.callback(None)
        return d

    def remote_deferredSecurity(self):
        d = defer.Deferred()
        d.addCallback(self.raiseSecurity)
        d.callback(None)
        return d

    def raiseJelly(self, results=None):
        raise JellyError("I'm jellyable!")

    def raiseSecurity(self, results=None):
        raise SecurityError("I'm secure!")