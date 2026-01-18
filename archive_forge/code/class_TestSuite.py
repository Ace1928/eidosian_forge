import logging
import unittest
import weakref
from typing import Dict, List
from .. import pyutils
class TestSuite(unittest.TestSuite):
    """I am an extended TestSuite with a visitor interface.
    This is primarily to allow filtering of tests - and suites or
    more in the future. An iterator of just tests wouldn't scale..."""

    def visit(self, visitor):
        """visit the composite. Visiting is depth-first.
        current callbacks are visitSuite and visitCase."""
        visitor.visitSuite(self)
        visitTests(self, visitor)

    def run(self, result):
        """Run the tests in the suite, discarding references after running."""
        tests = list(self)
        tests.reverse()
        self._tests = []
        stored_count = 0
        count_stored_tests = getattr(result, '_count_stored_tests', int)
        from breezy.tests import selftest_debug_flags
        notify = 'uncollected_cases' in selftest_debug_flags
        while tests:
            if result.shouldStop:
                self._tests = reversed(tests)
                break
            case = _run_and_collect_case(tests.pop(), result)()
            new_stored_count = count_stored_tests()
            if case is not None and isinstance(case, unittest.TestCase):
                if stored_count == new_stored_count and notify:
                    FailedCollectionCase(case).run(result)
                    new_stored_count = count_stored_tests()
            stored_count = new_stored_count
        return result