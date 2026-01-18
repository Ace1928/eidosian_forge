from twisted.internet.defer import (
from twisted.trial.unittest import SynchronousTestCase, TestCase
class CancellationTests(SynchronousTestCase):
    """
    Tests for cancellation of L{Deferred}s returned by L{inlineCallbacks}.
    For each of these tests, let:
        - C{G} be a generator decorated with C{inlineCallbacks}
        - C{D} be a L{Deferred} returned by C{G}
        - C{C} be a L{Deferred} awaited by C{G} with C{yield}
    """

    def setUp(self):
        """
        Set up the list of outstanding L{Deferred}s.
        """
        self.deferredsOutstanding = []

    def tearDown(self):
        """
        If any L{Deferred}s are still outstanding, fire them.
        """
        while self.deferredsOutstanding:
            self.deferredGotten()

    @inlineCallbacks
    def sampleInlineCB(self, getChildDeferred=None):
        """
        Generator for testing cascade cancelling cases.

        @param getChildDeferred: Some callable returning L{Deferred} that we
            awaiting (with C{yield})
        """
        if getChildDeferred is None:
            getChildDeferred = self.getDeferred
        try:
            x = (yield getChildDeferred())
        except UntranslatedError:
            raise TranslatedError()
        except DontFail as df:
            x = df.actualValue - 2
        returnValue(x + 1)

    def getDeferred(self):
        """
        A sample function that returns a L{Deferred} that can be fired on
        demand, by L{CancellationTests.deferredGotten}.

        @return: L{Deferred} that can be fired on demand.
        """
        self.deferredsOutstanding.append(Deferred())
        return self.deferredsOutstanding[-1]

    def deferredGotten(self, result=None):
        """
        Fire the L{Deferred} returned from the least-recent call to
        L{CancellationTests.getDeferred}.

        @param result: result object to be used when firing the L{Deferred}.
        """
        self.deferredsOutstanding.pop(0).callback(result)

    def test_cascadeCancellingOnCancel(self):
        """
        When C{D} cancelled, C{C} will be immediately cancelled too.
        """
        childResultHolder = ['FAILURE']

        def getChildDeferred():
            d = Deferred()

            def _eb(result):
                childResultHolder[0] = result.check(CancelledError)
                return result
            d.addErrback(_eb)
            return d
        d = self.sampleInlineCB(getChildDeferred=getChildDeferred)
        d.addErrback(lambda result: None)
        d.cancel()
        self.assertEqual(childResultHolder[0], CancelledError, 'no cascade cancelling occurs')

    def test_errbackCancelledErrorOnCancel(self):
        """
        When C{D} cancelled, CancelledError from C{C} will be errbacked
        through C{D}.
        """
        d = self.sampleInlineCB()
        d.cancel()
        self.assertRaises(CancelledError, self.failureResultOf(d).raiseException)

    def test_errorToErrorTranslation(self):
        """
        When C{D} is cancelled, and C raises a particular type of error, C{G}
        may catch that error at the point of yielding and translate it into
        a different error which may be received by application code.
        """

        def cancel(it):
            it.errback(UntranslatedError())
        a = Deferred(cancel)
        d = self.sampleInlineCB(lambda: a)
        d.cancel()
        self.assertRaises(TranslatedError, self.failureResultOf(d).raiseException)

    def test_errorToSuccessTranslation(self):
        """
        When C{D} is cancelled, and C{C} raises a particular type of error,
        C{G} may catch that error at the point of yielding and translate it
        into a result value which may be received by application code.
        """

        def cancel(it):
            it.errback(DontFail(4321))
        a = Deferred(cancel)
        d = self.sampleInlineCB(lambda: a)
        results = []
        d.addCallback(results.append)
        d.cancel()
        self.assertEquals(results, [4320])

    def test_asynchronousCancellation(self):
        """
        When C{D} is cancelled, it won't reach the callbacks added to it by
        application code until C{C} reaches the point in its callback chain
        where C{G} awaits it.  Otherwise, application code won't be able to
        track resource usage that C{D} may be using.
        """
        moreDeferred = Deferred()

        def deferMeMore(result):
            result.trap(CancelledError)
            return moreDeferred

        def deferMe():
            d = Deferred()
            d.addErrback(deferMeMore)
            return d
        d = self.sampleInlineCB(getChildDeferred=deferMe)
        d.cancel()
        self.assertNoResult(d)
        moreDeferred.callback(6543)
        self.assertEqual(self.successResultOf(d), 6544)