from testtools.matchers import Mismatch
from ._deferred import failure_content, on_deferred_result
def _got_failure(self, deferred, failure):
    deferred.addErrback(lambda _: None)
    return self._matcher.match(failure)