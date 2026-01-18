from twisted.internet import defer, error
from twisted.trial import unittest
from twisted.words.im import basesupport
class ClientMsgTests(unittest.TestCase):

    def makeUI(self):
        return DummyUI()

    def makeAccount(self):
        return DummyAccount('la', False, 'la', None, 'localhost', 6667)

    def test_connect(self):
        """
        Test that account.logOn works, and it calls the right callback when a
        connection is established.
        """
        account = self.makeAccount()
        ui = self.makeUI()
        d = account.logOn(ui)
        account.loginDeferred.callback(None)

        def check(result):
            self.assertFalse(account.loginHasFailed, "Login shouldn't have failed")
            self.assertTrue(account.loginCallbackCalled, 'We should be logged in')
        d.addCallback(check)
        return d

    def test_failedConnect(self):
        """
        Test that account.logOn works, and it calls the right callback when a
        connection is established.
        """
        account = self.makeAccount()
        ui = self.makeUI()
        d = account.logOn(ui)
        account.loginDeferred.errback(Exception())

        def err(reason):
            self.assertTrue(account.loginHasFailed, 'Login should have failed')
            self.assertFalse(account.loginCallbackCalled, "We shouldn't be logged in")
            self.assertTrue(not ui.clientRegistered, "Client shouldn't be registered in the UI")
        cb = lambda r: self.assertTrue(False, "Shouldn't get called back")
        d.addCallbacks(cb, err)
        return d

    def test_alreadyConnecting(self):
        """
        Test that it can fail sensibly when someone tried to connect before
        we did.
        """
        account = self.makeAccount()
        ui = self.makeUI()
        account.logOn(ui)
        self.assertRaises(error.ConnectError, account.logOn, ui)