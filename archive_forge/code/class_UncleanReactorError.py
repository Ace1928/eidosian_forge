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
class UncleanReactorError(Exception):
    """Raised when the reactor has junk in it."""

    def __init__(self, junk):
        Exception.__init__(self, 'The reactor still thinks it needs to do things. Close all connections, kill all processes and make sure all delayed calls have either fired or been cancelled:\n%s' % ''.join(map(self._get_junk_info, junk)))

    def _get_junk_info(self, junk):
        from twisted.internet.base import DelayedCall
        if isinstance(junk, DelayedCall):
            ret = str(junk)
        else:
            ret = repr(junk)
        return f'  {ret}\n'