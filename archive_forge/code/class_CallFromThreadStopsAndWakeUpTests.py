import os
import sys
import time
from unittest import skipIf
from twisted.internet import abstract, base, defer, error, interfaces, protocol, reactor
from twisted.internet.defer import Deferred, passthru
from twisted.internet.tcp import Connector
from twisted.python import util
from twisted.trial.unittest import TestCase
import %(reactor)s
from twisted.internet import reactor
class CallFromThreadStopsAndWakeUpTests(TestCase):

    @skipIf(not interfaces.IReactorThreads(reactor, None), 'Nothing to wake up for without thread support')
    def testWakeUp(self):
        d = Deferred()

        def wake():
            time.sleep(0.1)
            reactor.callFromThread(d.callback, None)
        reactor.callInThread(wake)
        return d

    def _stopCallFromThreadCallback(self):
        self.stopped = True

    def _callFromThreadCallback(self, d):
        reactor.callFromThread(self._callFromThreadCallback2, d)
        reactor.callLater(0, self._stopCallFromThreadCallback)

    def _callFromThreadCallback2(self, d):
        try:
            self.assertTrue(self.stopped)
        except BaseException:
            d.errback()
        else:
            d.callback(None)

    def testCallFromThreadStops(self):
        """
        Ensure that callFromThread from inside a callFromThread
        callback doesn't sit in an infinite loop and lets other
        things happen too.
        """
        self.stopped = False
        d = defer.Deferred()
        reactor.callFromThread(self._callFromThreadCallback, d)
        return d