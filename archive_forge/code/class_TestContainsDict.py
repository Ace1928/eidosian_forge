from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._dict import (
from testtools.tests.matchers.helpers import TestMatchersInterface
class TestContainsDict(TestCase, TestMatchersInterface):
    matches_matcher = ContainsDict({'foo': Equals('bar'), 'baz': Not(Equals('qux'))})
    matches_matches = [{'foo': 'bar', 'baz': None}, {'foo': 'bar', 'baz': 'quux'}, {'foo': 'bar', 'baz': 'quux', 'cat': 'dog'}]
    matches_mismatches = [{}, {'foo': 'bar', 'baz': 'qux'}, {'foo': 'bop', 'baz': 'qux'}, {'foo': 'bar', 'cat': 'dog'}, {'foo': 'bar'}]
    str_examples = [("ContainsDict({{'baz': {}, 'foo': {}}})".format(Not(Equals('qux')), Equals('bar')), matches_matcher)]
    describe_examples = [("Missing: {\n  'baz': Not(Equals('qux')),\n  'foo': Equals('bar'),\n}", {}, matches_matcher), ("Differences: {\n  'baz': 'qux' matches Equals('qux'),\n}", {'foo': 'bar', 'baz': 'qux'}, matches_matcher), ("Differences: {\n  'baz': 'qux' matches Equals('qux'),\n  'foo': 'bop' != 'bar',\n}", {'foo': 'bop', 'baz': 'qux'}, matches_matcher), ("Missing: {\n  'baz': Not(Equals('qux')),\n}", {'foo': 'bar', 'cat': 'dog'}, matches_matcher)]