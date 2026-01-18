from zope.interface import implementer
from twisted.internet import defer, interfaces
from twisted.protocols import basic
from twisted.python.failure import Failure
from twisted.spread import pb
class LocalAsRemote:
    """
    A class useful for emulating the effects of remote behavior locally.
    """
    reportAllTracebacks = 1

    def callRemote(self, name, *args, **kw):
        """
        Call a specially-designated local method.

        self.callRemote('x') will first try to invoke a method named
        sync_x and return its result (which should probably be a
        Deferred).  Second, it will look for a method called async_x,
        which will be called and then have its result (or Failure)
        automatically wrapped in a Deferred.
        """
        if hasattr(self, 'sync_' + name):
            return getattr(self, 'sync_' + name)(*args, **kw)
        try:
            method = getattr(self, 'async_' + name)
            return defer.succeed(method(*args, **kw))
        except BaseException:
            f = Failure()
            if self.reportAllTracebacks:
                f.printTraceback()
            return defer.fail(f)

    def remoteMethod(self, name):
        return LocalMethod(self, name)