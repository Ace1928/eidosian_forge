import inspect
import sys
from typing import List
from unittest import skipIf
from zope.interface import directlyProvides
import twisted.mail._pop3client
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.internet.testing import StringTransport
from twisted.mail.pop3 import (
from twisted.mail.test import pop3testserver
from twisted.protocols import basic, loopback
from twisted.python import log
from twisted.trial.unittest import TestCase
class POP3TimeoutTests(POP3HelperMixin, TestCase):

    def testTimeout(self):

        def login():
            d = self.client.login('test', 'twisted')
            d.addCallback(loggedIn)
            d.addErrback(timedOut)
            return d

        def loggedIn(result):
            self.fail('Successfully logged in!?  Impossible!')

        def timedOut(failure):
            failure.trap(error.TimeoutError)
            self._cbStopClient(None)

        def quit():
            return self.client.quit()
        self.client.timeout = 0.01
        pop3testserver.TIMEOUT_RESPONSE = True
        methods = [login, quit]
        map(self.connected.addCallback, map(strip, methods))
        self.connected.addCallback(self._cbStopClient)
        self.connected.addErrback(self._ebGeneral)
        return self.loopback()