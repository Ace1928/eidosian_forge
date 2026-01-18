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
class _ErrorObserver(Fixture):
    """Capture errors logged while fixture is active."""

    def __init__(self, error_observer):
        super().__init__()
        self._error_observer = error_observer

    def _setUp(self):
        self.useFixture(_TwistedLogObservers([self._error_observer.gotEvent]))

    def flush_logged_errors(self, *error_types):
        """Clear errors of the given types from the logs.

        If no errors provided, clear all errors.

        :return: An iterable of errors removed from the logs.
        """
        return self._error_observer.flushErrors(*error_types)