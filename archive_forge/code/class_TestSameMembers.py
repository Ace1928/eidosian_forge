import re
from testtools import TestCase
from testtools.compat import (
from testtools.matchers._basic import (
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
class TestSameMembers(TestCase, TestMatchersInterface):
    matches_matcher = SameMembers([1, 1, 2, 3, {'foo': 'bar'}])
    matches_matches = [[1, 1, 2, 3, {'foo': 'bar'}], [3, {'foo': 'bar'}, 1, 2, 1], [3, 2, 1, {'foo': 'bar'}, 1], (2, {'foo': 'bar'}, 3, 1, 1)]
    matches_mismatches = [{1, 2, 3}, [1, 1, 2, 3, 5], [1, 2, 3, {'foo': 'bar'}], 'foo']
    describe_examples = [("elements differ:\nreference = ['apple', 'orange', 'canteloupe', 'watermelon', 'lemon', 'banana']\nactual    = ['orange', 'apple', 'banana', 'sparrow', 'lemon', 'canteloupe']\n: \nmissing:    ['watermelon']\nextra:      ['sparrow']", ['orange', 'apple', 'banana', 'sparrow', 'lemon', 'canteloupe'], SameMembers(['apple', 'orange', 'canteloupe', 'watermelon', 'lemon', 'banana']))]
    str_examples = [('SameMembers([1, 2, 3])', SameMembers([1, 2, 3]))]