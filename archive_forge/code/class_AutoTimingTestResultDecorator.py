import csv
import datetime
import testtools
from testtools import StreamResult
from testtools.content import TracebackContent, text_content
import iso8601
import subunit
class AutoTimingTestResultDecorator(HookedTestResultDecorator):
    """Decorate a TestResult to add time events to a test run.

    By default this will cause a time event before every test event,
    but if explicit time data is being provided by the test run, then
    this decorator will turn itself off to prevent causing confusion.
    """

    def __init__(self, decorated):
        self._time = None
        super().__init__(decorated)

    def _before_event(self):
        time = self._time
        if time is not None:
            return
        time = datetime.datetime.now(tz=iso8601.UTC)
        self.decorated.time(time)

    def progress(self, offset, whence):
        return self.decorated.progress(offset, whence)

    @property
    def shouldStop(self):
        return self.decorated.shouldStop

    def time(self, a_datetime):
        """Provide a timestamp for the current test activity.

        :param a_datetime: If None, automatically add timestamps before every
            event (this is the default behaviour if time() is not called at
            all).  If not None, pass the provided time onto the decorated
            result object and disable automatic timestamps.
        """
        self._time = a_datetime
        return self.decorated.time(a_datetime)