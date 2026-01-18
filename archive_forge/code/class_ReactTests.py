from twisted.internet import defer, error, interfaces, reactor, task
from twisted.internet.main import installReactor
from twisted.internet.test.modulehelpers import NoReactor
from twisted.trial import unittest
from twisted.python import failure
class ReactTests(unittest.SynchronousTestCase):
    """
    Tests for L{twisted.internet.task.react}.
    """

    def test_runsUntilAsyncCallback(self):
        """
        L{task.react} runs the reactor until the L{Deferred} returned by the
        function it is passed is called back, then stops it.
        """
        timePassed = []

        def main(reactor):
            finished = defer.Deferred()
            reactor.callLater(1, timePassed.append, True)
            reactor.callLater(2, finished.callback, None)
            return finished
        r = _FakeReactor()
        exitError = self.assertRaises(SystemExit, task.react, main, _reactor=r)
        self.assertEqual(0, exitError.code)
        self.assertEqual(timePassed, [True])
        self.assertEqual(r.seconds(), 2)

    def test_runsUntilSyncCallback(self):
        """
        L{task.react} returns quickly if the L{Deferred} returned by the
        function it is passed has already been called back at the time it is
        returned.
        """

        def main(reactor):
            return defer.succeed(None)
        r = _FakeReactor()
        exitError = self.assertRaises(SystemExit, task.react, main, _reactor=r)
        self.assertEqual(0, exitError.code)
        self.assertEqual(r.seconds(), 0)

    def test_runsUntilAsyncErrback(self):
        """
        L{task.react} runs the reactor until the L{defer.Deferred} returned by
        the function it is passed is errbacked, then it stops the reactor and
        reports the error.
        """

        class ExpectedException(Exception):
            pass

        def main(reactor):
            finished = defer.Deferred()
            reactor.callLater(1, finished.errback, ExpectedException())
            return finished
        r = _FakeReactor()
        exitError = self.assertRaises(SystemExit, task.react, main, _reactor=r)
        self.assertEqual(1, exitError.code)
        errors = self.flushLoggedErrors(ExpectedException)
        self.assertEqual(len(errors), 1)

    def test_runsUntilSyncErrback(self):
        """
        L{task.react} returns quickly if the L{defer.Deferred} returned by the
        function it is passed has already been errbacked at the time it is
        returned.
        """

        class ExpectedException(Exception):
            pass

        def main(reactor):
            return defer.fail(ExpectedException())
        r = _FakeReactor()
        exitError = self.assertRaises(SystemExit, task.react, main, _reactor=r)
        self.assertEqual(1, exitError.code)
        self.assertEqual(r.seconds(), 0)
        errors = self.flushLoggedErrors(ExpectedException)
        self.assertEqual(len(errors), 1)

    def test_singleStopCallback(self):
        """
        L{task.react} doesn't try to stop the reactor if the L{defer.Deferred}
        the function it is passed is called back after the reactor has already
        been stopped.
        """

        def main(reactor):
            reactor.callLater(1, reactor.stop)
            finished = defer.Deferred()
            reactor.addSystemEventTrigger('during', 'shutdown', finished.callback, None)
            return finished
        r = _FakeReactor()
        exitError = self.assertRaises(SystemExit, task.react, main, _reactor=r)
        self.assertEqual(r.seconds(), 1)
        self.assertEqual(0, exitError.code)

    def test_singleStopErrback(self):
        """
        L{task.react} doesn't try to stop the reactor if the L{defer.Deferred}
        the function it is passed is errbacked after the reactor has already
        been stopped.
        """

        class ExpectedException(Exception):
            pass

        def main(reactor):
            reactor.callLater(1, reactor.stop)
            finished = defer.Deferred()
            reactor.addSystemEventTrigger('during', 'shutdown', finished.errback, ExpectedException())
            return finished
        r = _FakeReactor()
        exitError = self.assertRaises(SystemExit, task.react, main, _reactor=r)
        self.assertEqual(1, exitError.code)
        self.assertEqual(r.seconds(), 1)
        errors = self.flushLoggedErrors(ExpectedException)
        self.assertEqual(len(errors), 1)

    def test_arguments(self):
        """
        L{task.react} passes the elements of the list it is passed as
        positional arguments to the function it is passed.
        """
        args = []

        def main(reactor, x, y, z):
            args.extend((x, y, z))
            return defer.succeed(None)
        r = _FakeReactor()
        exitError = self.assertRaises(SystemExit, task.react, main, [1, 2, 3], _reactor=r)
        self.assertEqual(0, exitError.code)
        self.assertEqual(args, [1, 2, 3])

    def test_defaultReactor(self):
        """
        L{twisted.internet.reactor} is used if no reactor argument is passed to
        L{task.react}.
        """

        def main(reactor):
            self.passedReactor = reactor
            return defer.succeed(None)
        reactor = _FakeReactor()
        with NoReactor():
            installReactor(reactor)
            exitError = self.assertRaises(SystemExit, task.react, main, [])
            self.assertEqual(0, exitError.code)
        self.assertIs(reactor, self.passedReactor)

    def test_exitWithDefinedCode(self):
        """
        L{task.react} forwards the exit code specified by the C{SystemExit}
        error returned by the passed function, if any.
        """

        def main(reactor):
            return defer.fail(SystemExit(23))
        r = _FakeReactor()
        exitError = self.assertRaises(SystemExit, task.react, main, [], _reactor=r)
        self.assertEqual(23, exitError.code)

    def test_synchronousStop(self):
        """
        L{task.react} handles when the reactor is stopped just before the
        returned L{Deferred} fires.
        """

        def main(reactor):
            d = defer.Deferred()

            def stop():
                reactor.stop()
                d.callback(None)
            reactor.callWhenRunning(stop)
            return d
        r = _FakeReactor()
        exitError = self.assertRaises(SystemExit, task.react, main, [], _reactor=r)
        self.assertEqual(0, exitError.code)

    def test_asynchronousStop(self):
        """
        L{task.react} handles when the reactor is stopped and the
        returned L{Deferred} doesn't fire.
        """

        def main(reactor):
            reactor.callLater(1, reactor.stop)
            return defer.Deferred()
        r = _FakeReactor()
        exitError = self.assertRaises(SystemExit, task.react, main, [], _reactor=r)
        self.assertEqual(0, exitError.code)