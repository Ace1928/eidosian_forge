from testtools.matchers import Mismatch
from ._deferred import failure_content, on_deferred_result
@staticmethod
def _got_no_result(deferred):
    return Mismatch('Failure result expected on %r, found no result instead' % (deferred,))