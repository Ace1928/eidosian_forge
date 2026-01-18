import warnings
from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._warnings import Warnings, IsDeprecated, WarningMessage
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
class TestWarningMessageFilenameInterface(TestCase, TestMatchersInterface):
    """
    Tests for `testtools.matchers._warnings.WarningMessage`.

    In particular matching the ``filename``.
    """
    matches_matcher = WarningMessage(category_type=DeprecationWarning, filename=Equals('a'))
    warning_foo = make_warning_message('foo', DeprecationWarning, filename='a')
    warning_bar = make_warning_message('bar', DeprecationWarning, filename='b')
    matches_matches = [warning_foo]
    matches_mismatches = [warning_bar]
    str_examples = []
    describe_examples = []