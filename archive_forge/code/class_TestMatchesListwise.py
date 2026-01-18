import doctest
import io
import re
import sys
from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._datastructures import (
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
class TestMatchesListwise(TestCase):
    run_tests_with = FullStackRunTest

    def test_docstring(self):
        failure_count, output = run_doctest(MatchesListwise, 'MatchesListwise')
        if failure_count:
            self.fail('Doctest failed with %s' % output)