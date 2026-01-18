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
def _run_deferred(self):
    """Run the test, assuming everything in it is Deferred-returning.

        This should return a Deferred that fires with True if the test was
        successful and False if the test was not successful.  It should *not*
        call addSuccess on the result, because there's reactor clean up that
        we needs to be done afterwards.
        """
    fails = []

    def fail_if_exception_caught(exception_caught):
        if self.exception_caught == exception_caught:
            fails.append(None)

    def clean_up(ignored=None):
        """Run the cleanups."""
        d = self._run_cleanups()

        def clean_up_done(result):
            if result is not None:
                self._exceptions.append(result)
                fails.append(None)
        return d.addCallback(clean_up_done)

    def set_up_done(exception_caught):
        """Set up is done, either clean up or run the test."""
        if self.exception_caught == exception_caught:
            fails.append(None)
            return clean_up()
        else:
            d = self._run_user(self.case._run_test_method, self.result)
            d.addCallback(fail_if_exception_caught)
            d.addBoth(tear_down)
            return d

    def tear_down(ignored):
        d = self._run_user(self.case._run_teardown, self.result)
        d.addCallback(fail_if_exception_caught)
        d.addBoth(clean_up)
        return d

    def force_failure(ignored):
        if getattr(self.case, 'force_failure', None):
            d = self._run_user(_raise_force_fail_error)
            d.addCallback(fails.append)
            return d
    d = self._run_user(self.case._run_setup, self.result)
    d.addCallback(set_up_done)
    d.addBoth(force_failure)
    d.addBoth(lambda ignored: len(fails) == 0)
    return d