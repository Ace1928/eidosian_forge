from testtools.content import TracebackContent
from testtools.matchers import (
from ._helpers import NeedsTwistedTestCase
from testtools.twistedsupport import has_no_result, failed, succeeded
from twisted.internet import defer
from twisted.python.failure import Failure
class FailureResultTests(NeedsTwistedTestCase):

    def match(self, matcher, value):
        return failed(matcher).match(value)

    def test_failure_passes(self):
        fail = make_failure(RuntimeError('arbitrary failure'))
        deferred = defer.fail(fail)
        self.assertThat(self.match(Is(fail), deferred), Is(None))

    def test_different_failure_fails(self):
        fail = make_failure(RuntimeError('arbitrary failure'))
        deferred = defer.fail(fail)
        matcher = Is(None)
        mismatch = matcher.match(fail)
        self.assertThat(self.match(matcher, deferred), mismatches(Equals(mismatch.describe()), Equals(mismatch.get_details())))

    def test_success_fails(self):
        result = object()
        deferred = defer.succeed(result)
        matcher = Is(None)
        self.assertThat(self.match(matcher, deferred), mismatches(Equals('Failure result expected on %r, found success result (%r) instead' % (deferred, result))))

    def test_no_result_fails(self):
        deferred = defer.Deferred()
        matcher = Is(None)
        self.assertThat(self.match(matcher, deferred), mismatches(Equals('Failure result expected on %r, found no result instead' % (deferred,))))