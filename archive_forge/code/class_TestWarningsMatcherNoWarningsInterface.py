import warnings
from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._warnings import Warnings, IsDeprecated, WarningMessage
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
class TestWarningsMatcherNoWarningsInterface(TestCase, TestMatchersInterface):
    """
    Tests for `testtools.matchers._warnings.Warnings`.

    Specifically with the optional matcher argument matching that there were no
    warnings.
    """
    matches_matcher = Warnings(warnings_matcher=HasLength(0))

    def nowarning_func():
        pass

    def warning_func():
        warnings.warn('warning_func is deprecated', DeprecationWarning, 2)
    matches_matches = [nowarning_func]
    matches_mismatches = [warning_func]
    str_examples = []
    describe_examples = []