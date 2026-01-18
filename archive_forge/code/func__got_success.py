from testtools.matchers import Mismatch
from ._deferred import failure_content, on_deferred_result
@staticmethod
def _got_success(deferred, success):
    return Mismatch('Failure result expected on %r, found success result (%r) instead' % (deferred, success))