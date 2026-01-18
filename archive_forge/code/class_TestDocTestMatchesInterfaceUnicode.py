import doctest
from testtools import TestCase
from testtools.compat import (
from testtools.matchers._doctest import DocTestMatches
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
class TestDocTestMatchesInterfaceUnicode(TestCase, TestMatchersInterface):
    matches_matcher = DocTestMatches('§...', doctest.ELLIPSIS)
    matches_matches = ['§', '§ more\n']
    matches_mismatches = ['\\xa7', 'more §', '\n§']
    str_examples = [('DocTestMatches({!r})'.format('§\n'), DocTestMatches('§'))]
    describe_examples = [('Expected:\n    §\nGot:\n    a\n', 'a', DocTestMatches('§', doctest.ELLIPSIS))]