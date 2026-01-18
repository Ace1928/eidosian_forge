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
def _got_user_failure(self, failure, tb_label='traceback'):
    """We got a failure from user code."""
    return self._got_user_exception((failure.type, failure.value, failure.getTracebackObject()), tb_label=tb_label)