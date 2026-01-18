from zope.interface import implementer
from twisted.internet import defer, interfaces
from twisted.protocols import basic
from twisted.python.failure import Failure
from twisted.spread import pb
class LocalAsyncForwarder:
    """
    A class useful for forwarding a locally-defined interface.
    """

    def __init__(self, forwarded, interfaceClass, failWhenNotImplemented=0):
        assert interfaceClass.providedBy(forwarded)
        self.forwarded = forwarded
        self.interfaceClass = interfaceClass
        self.failWhenNotImplemented = failWhenNotImplemented

    def _callMethod(self, method, *args, **kw):
        return getattr(self.forwarded, method)(*args, **kw)

    def callRemote(self, method, *args, **kw):
        if self.interfaceClass.queryDescriptionFor(method):
            result = defer.maybeDeferred(self._callMethod, method, *args, **kw)
            return result
        elif self.failWhenNotImplemented:
            return defer.fail(Failure(NotImplementedError, 'No Such Method in Interface: %s' % method))
        else:
            return defer.succeed(None)