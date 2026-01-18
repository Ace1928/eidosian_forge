from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._dict import (
from testtools.tests.matchers.helpers import TestMatchersInterface
class TestSubDictOf(TestCase, TestMatchersInterface):
    matches_matcher = _SubDictOf({'foo': 'bar', 'baz': 'qux'})
    matches_matches = [{'foo': 'bar', 'baz': 'qux'}, {'foo': 'bar'}]
    matches_mismatches = [{'foo': 'bar', 'baz': 'qux', 'cat': 'dog'}, {'foo': 'bar', 'cat': 'dog'}]
    str_examples = []
    describe_examples = []