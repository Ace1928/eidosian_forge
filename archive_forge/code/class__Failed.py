from testtools.matchers import Mismatch
from ._deferred import failure_content, on_deferred_result
class _Failed:
    """Matches a Deferred that has failed."""

    def __init__(self, matcher):
        self._matcher = matcher

    def _got_failure(self, deferred, failure):
        deferred.addErrback(lambda _: None)
        return self._matcher.match(failure)

    @staticmethod
    def _got_success(deferred, success):
        return Mismatch('Failure result expected on %r, found success result (%r) instead' % (deferred, success))

    @staticmethod
    def _got_no_result(deferred):
        return Mismatch('Failure result expected on %r, found no result instead' % (deferred,))

    def match(self, deferred):
        return on_deferred_result(deferred, on_success=self._got_success, on_failure=self._got_failure, on_no_result=self._got_no_result)