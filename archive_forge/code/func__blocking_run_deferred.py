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
def _blocking_run_deferred(self, spinner):
    try:
        return trap_unhandled_errors(spinner.run, self._timeout, self._run_deferred)
    except NoResultError:
        self._got_user_exception(sys.exc_info())
        self.result.stop()
        return (False, [])
    except TimeoutError:
        self._log_user_exception(TimeoutError(self.case, self._timeout))
        return (False, [])