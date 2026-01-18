from testtools.content import TracebackContent
from testtools.matchers import (
from ._helpers import NeedsTwistedTestCase
from testtools.twistedsupport import has_no_result, failed, succeeded
from twisted.internet import defer
from twisted.python.failure import Failure
class NoResultTests(NeedsTwistedTestCase):
    """Tests for ``has_no_result``."""

    def match(self, thing):
        return has_no_result().match(thing)

    def test_unfired_matches(self):
        self.assertThat(self.match(defer.Deferred()), Is(None))

    def test_succeeded_does_no_match(self):
        result = object()
        deferred = defer.succeed(result)
        mismatch = self.match(deferred)
        self.assertThat(mismatch, mismatches(Equals('No result expected on %r, found %r instead' % (deferred, result))))

    def test_failed_does_not_match(self):
        fail = make_failure(RuntimeError('arbitrary failure'))
        deferred = defer.fail(fail)
        self.addCleanup(deferred.addErrback, lambda _: None)
        mismatch = self.match(deferred)
        self.assertThat(mismatch, mismatches(Equals('No result expected on %r, found %r instead' % (deferred, fail))))

    def test_success_after_assertion(self):
        deferred = defer.Deferred()
        self.assertThat(deferred, has_no_result())
        results = []
        deferred.addCallback(results.append)
        marker = object()
        deferred.callback(marker)
        self.assertThat(results, Equals([marker]))

    def test_failure_after_assertion(self):
        deferred = defer.Deferred()
        self.assertThat(deferred, has_no_result())
        results = []
        deferred.addErrback(results.append)
        fail = make_failure(RuntimeError('arbitrary failure'))
        deferred.errback(fail)
        self.assertThat(results, Equals([fail]))