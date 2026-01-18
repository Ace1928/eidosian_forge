from twisted.internet import defer, error
from twisted.trial import unittest
from twisted.words.im import basesupport
class DummyAccount(basesupport.AbstractAccount):
    """
    An account object that will do nothing when asked to start to log on.
    """
    loginHasFailed = False
    loginCallbackCalled = False

    def _startLogOn(self, *args):
        """
        Set self.loginDeferred to the same as the deferred returned, allowing a
        testcase to .callback or .errback.

        @return: A deferred.
        """
        self.loginDeferred = defer.Deferred()
        return self.loginDeferred

    def _loginFailed(self, result):
        self.loginHasFailed = True
        return basesupport.AbstractAccount._loginFailed(self, result)

    def _cb_logOn(self, result):
        self.loginCallbackCalled = True
        return basesupport.AbstractAccount._cb_logOn(self, result)