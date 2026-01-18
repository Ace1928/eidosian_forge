from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._higherorder import (
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
class TestAfterPreprocessing(TestCase, TestMatchersInterface):

    def parity(x):
        return x % 2
    matches_matcher = AfterPreprocessing(parity, Equals(1))
    matches_matches = [3, 5]
    matches_mismatches = [2]
    str_examples = [('AfterPreprocessing(<function parity>, Equals(1))', AfterPreprocessing(parity, Equals(1)))]
    describe_examples = [('0 != 1: after <function parity> on 2', 2, AfterPreprocessing(parity, Equals(1))), ('0 != 1', 2, AfterPreprocessing(parity, Equals(1), annotate=False))]