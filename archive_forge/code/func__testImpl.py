from io import StringIO
from twisted.internet import defer
from twisted.python import log
from twisted.python.reflect import qual
from twisted.spread import flavors, jelly, pb
from twisted.test.iosim import connectedServerAndClient
from twisted.trial import unittest
def _testImpl(self, method, expected, eb, exc=None):
    """
        Call the given remote method and attach the given errback to the
        resulting Deferred.  If C{exc} is not None, also assert that one
        exception of that type was logged.
        """
    rootDeferred = self.clientFactory.getRootObject()

    def gotRootObj(obj):
        failureDeferred = self._addFailingCallbacks(obj.callRemote(method), expected, eb)
        if exc is not None:

            def gotFailure(err):
                self.assertEqual(len(self.flushLoggedErrors(exc)), 1)
                return err
            failureDeferred.addBoth(gotFailure)
        return failureDeferred
    rootDeferred.addCallback(gotRootObj)
    self.pump.flush()