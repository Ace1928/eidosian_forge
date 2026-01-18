from testtools.matchers import Mismatch
from ._deferred import failure_content, on_deferred_result
class _Succeeded:
    """Matches a Deferred that has fired successfully."""

    def __init__(self, matcher):
        """Construct a ``_Succeeded`` matcher."""
        self._matcher = matcher

    @staticmethod
    def _got_failure(deferred, failure):
        deferred.addErrback(lambda _: None)
        return Mismatch('Success result expected on %r, found failure result instead: %r' % (deferred, failure), {'traceback': failure_content(failure)})

    @staticmethod
    def _got_no_result(deferred):
        return Mismatch('Success result expected on %r, found no result instead' % (deferred,))

    def match(self, deferred):
        """Match against the successful result of ``deferred``."""
        return on_deferred_result(deferred, on_success=lambda _, value: self._matcher.match(value), on_failure=self._got_failure, on_no_result=self._got_no_result)