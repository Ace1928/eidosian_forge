import re
from testtools import TestCase
from testtools.compat import (
from testtools.matchers._basic import (
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
class TestIsInstanceInterface(TestCase, TestMatchersInterface):

    class Foo:
        pass
    matches_matcher = IsInstance(Foo)
    matches_matches = [Foo()]
    matches_mismatches = [object(), 1, Foo]
    str_examples = [('IsInstance(str)', IsInstance(str)), ('IsInstance(str, int)', IsInstance(str, int))]
    describe_examples = [("'foo' is not an instance of int", 'foo', IsInstance(int)), ("'foo' is not an instance of any of (int, type)", 'foo', IsInstance(int, type))]