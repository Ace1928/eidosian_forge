import warnings
from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._warnings import Warnings, IsDeprecated, WarningMessage
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
class TestWarningMessageLineNumberInterface(TestCase, TestMatchersInterface):
    """
    Tests for `testtools.matchers._warnings.WarningMessage`.

    In particular matching the ``lineno``.
    """
    matches_matcher = WarningMessage(category_type=DeprecationWarning, lineno=Equals(42))
    warning_foo = make_warning_message('foo', DeprecationWarning, lineno=42)
    warning_bar = make_warning_message('bar', DeprecationWarning, lineno=21)
    matches_matches = [warning_foo]
    matches_mismatches = [warning_bar]
    str_examples = []
    describe_examples = []