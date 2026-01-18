import warnings
from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._warnings import Warnings, IsDeprecated, WarningMessage
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
class TestWarningMessageCategoryTypeInterface(TestCase, TestMatchersInterface):
    """
    Tests for `testtools.matchers._warnings.WarningMessage`.

    In particular matching the ``category_type``.
    """
    matches_matcher = WarningMessage(category_type=DeprecationWarning)
    warning_foo = make_warning_message('foo', DeprecationWarning)
    warning_bar = make_warning_message('bar', SyntaxWarning)
    warning_base = make_warning_message('base', Warning)
    matches_matches = [warning_foo]
    matches_mismatches = [warning_bar, warning_base]
    str_examples = []
    describe_examples = []