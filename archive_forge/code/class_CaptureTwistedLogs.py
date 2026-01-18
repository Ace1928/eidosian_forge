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
class CaptureTwistedLogs(Fixture):
    """Capture all the Twisted logs and add them as a detail.

    Much of the time, you won't need to use this directly, as
    :py:class:`AsynchronousDeferredRunTest` captures Twisted logs when the
    ``store_twisted_logs`` is set to ``True`` (which it is by default).

    However, if you want to do custom processing of Twisted's logs, then this
    class can be useful.

    For example::

        class TwistedTests(TestCase):
            run_tests_with(
                partial(AsynchronousDeferredRunTest, store_twisted_logs=False))

            def setUp(self):
                super(TwistedTests, self).setUp()
                twisted_logs = self.useFixture(CaptureTwistedLogs())
                # ... do something with twisted_logs ...
    """
    LOG_DETAIL_NAME = 'twisted-log'

    def _setUp(self):
        logs = io.StringIO()
        full_observer = log.FileLogObserver(logs)
        self.useFixture(_TwistedLogObservers([full_observer.emit]))
        self.addDetail(self.LOG_DETAIL_NAME, Content(UTF8_TEXT, lambda: [logs.getvalue().encode('utf-8')]))