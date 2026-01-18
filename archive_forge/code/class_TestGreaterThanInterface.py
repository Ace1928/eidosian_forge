import re
from testtools import TestCase
from testtools.compat import (
from testtools.matchers._basic import (
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
class TestGreaterThanInterface(TestCase, TestMatchersInterface):
    matches_matcher = GreaterThan(4)
    matches_matches = [5, 8]
    matches_mismatches = [-2, 0, 4]
    str_examples = [('GreaterThan(12)', GreaterThan(12))]
    describe_examples = [('4 <= 5', 4, GreaterThan(5)), ('4 <= 4', 4, GreaterThan(4))]