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
class _TwistedLogObservers(Fixture):
    """Temporarily add Twisted log observers."""

    def __init__(self, observers):
        super().__init__()
        self._observers = observers
        self._log_publisher = log.theLogPublisher

    def _setUp(self):
        for observer in self._observers:
            self._log_publisher.addObserver(observer)
            self.addCleanup(self._log_publisher.removeObserver, observer)