import re
from testtools import TestCase
from testtools.compat import (
from testtools.matchers._basic import (
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
class TestEqualsInterface(TestCase, TestMatchersInterface):
    matches_matcher = Equals(1)
    matches_matches = [1]
    matches_mismatches = [2]
    str_examples = [('Equals(1)', Equals(1)), ("Equals('1')", Equals('1'))]
    describe_examples = [('2 != 1', 2, Equals(1)), ("!=:\nreference = 'abcdefghijklmnopqrstuvwxyz0123456789'\nactual    = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'\n", 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', Equals('abcdefghijklmnopqrstuvwxyz0123456789'))]