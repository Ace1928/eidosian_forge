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
def _run_user(self, function, *args):
    """Run a user-supplied function.

        This just makes sure that it returns a Deferred, regardless of how the
        user wrote it.
        """
    d = defer.maybeDeferred(function, *args)
    return d.addErrback(self._got_user_failure)