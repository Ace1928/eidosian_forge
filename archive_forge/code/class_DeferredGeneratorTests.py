import traceback
from twisted.internet import defer, reactor, task
from twisted.internet.defer import (
from twisted.python.util import runWithWarningsSuppressed
from twisted.trial import unittest
from twisted.trial.util import suppress as SUPPRESS
class DeferredGeneratorTests(BaseDefgenTests, unittest.TestCase):

    @deprecatedDeferredGenerator
    def _genBasics(self):
        x = waitForDeferred(getThing())
        yield x
        x = x.getResult()
        self.assertEqual(x, 'hi')
        ow = waitForDeferred(getOwie())
        yield ow
        try:
            ow.getResult()
        except ZeroDivisionError as e:
            self.assertEqual(str(e), 'OMG')
        yield 'WOOSH'
        return

    @deprecatedDeferredGenerator
    def _genBuggy(self):
        yield waitForDeferred(getThing())
        1 // 0

    @deprecatedDeferredGenerator
    def _genNothing(self):
        if False:
            yield 1

    @deprecatedDeferredGenerator
    def _genHandledTerminalFailure(self):
        x = waitForDeferred(defer.fail(TerminalException('Handled Terminal Failure')))
        yield x
        try:
            x.getResult()
        except TerminalException:
            pass

    @deprecatedDeferredGenerator
    def _genHandledTerminalAsyncFailure(self, d):
        x = waitForDeferred(d)
        yield x
        try:
            x.getResult()
        except TerminalException:
            pass

    def _genStackUsage(self):
        for x in range(5000):
            x = waitForDeferred(defer.succeed(1))
            yield x
            x = x.getResult()
        yield 0
    _genStackUsage = deprecatedDeferredGenerator(_genStackUsage)

    def _genStackUsage2(self):
        for x in range(5000):
            yield 1
        yield 0
    _genStackUsage2 = deprecatedDeferredGenerator(_genStackUsage2)

    def testDeferredYielding(self):
        """
        Ensure that yielding a Deferred directly is trapped as an
        error.
        """

        def _genDeferred():
            yield getThing()
        _genDeferred = deprecatedDeferredGenerator(_genDeferred)
        return self.assertFailure(_genDeferred(), TypeError)
    suppress = [SUPPRESS(message='twisted.internet.defer.waitForDeferred was deprecated')]