import io
import warnings
import sys
from fixtures import CompoundFixture, Fixture
from testtools.content import Content, text_content
from testtools.content_type import UTF8_TEXT
from testtools.runtest import RunTest, _raise_force_fail_error
from ._deferred import extract_result
from ._spinner import (
from twisted.internet import defer
from twisted.python import log
@defer.inlineCallbacks
def _run_cleanups(self):
    """Run the cleanups on the test case.

        We expect that the cleanups on the test case can also return
        asynchronous Deferreds.  As such, we take the responsibility for
        running the cleanups, rather than letting TestCase do it.
        """
    last_exception = None
    while self.case._cleanups:
        f, args, kwargs = self.case._cleanups.pop()
        d = defer.maybeDeferred(f, *args, **kwargs)
        try:
            yield d
        except Exception:
            exc_info = sys.exc_info()
            self.case._report_traceback(exc_info)
            last_exception = exc_info[1]
    defer.returnValue(last_exception)