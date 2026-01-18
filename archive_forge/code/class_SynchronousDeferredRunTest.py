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
class SynchronousDeferredRunTest(_DeferredRunTest):
    """Runner for tests that return synchronous Deferreds.

    This runner doesn't touch the reactor at all. It assumes that tests return
    Deferreds that have already fired.
    """

    def _run_user(self, function, *args):
        d = defer.maybeDeferred(function, *args)
        d.addErrback(self._got_user_failure)
        result = extract_result(d)
        return result