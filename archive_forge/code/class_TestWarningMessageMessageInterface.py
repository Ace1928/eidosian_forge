import warnings
from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._warnings import Warnings, IsDeprecated, WarningMessage
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
class TestWarningMessageMessageInterface(TestCase, TestMatchersInterface):
    """
    Tests for `testtools.matchers._warnings.WarningMessage`.

    In particular matching the ``message``.
    """
    matches_matcher = WarningMessage(category_type=DeprecationWarning, message=Equals('foo'))
    warning_foo = make_warning_message('foo', DeprecationWarning)
    warning_bar = make_warning_message('bar', DeprecationWarning)
    matches_matches = [warning_foo]
    matches_mismatches = [warning_bar]
    str_examples = []
    describe_examples = []