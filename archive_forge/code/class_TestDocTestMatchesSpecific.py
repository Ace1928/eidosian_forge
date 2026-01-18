import doctest
from testtools import TestCase
from testtools.compat import (
from testtools.matchers._doctest import DocTestMatches
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
class TestDocTestMatchesSpecific(TestCase):
    run_tests_with = FullStackRunTest

    def test___init__simple(self):
        matcher = DocTestMatches('foo')
        self.assertEqual('foo\n', matcher.want)

    def test___init__flags(self):
        matcher = DocTestMatches('bar\n', doctest.ELLIPSIS)
        self.assertEqual('bar\n', matcher.want)
        self.assertEqual(doctest.ELLIPSIS, matcher.flags)

    def test_describe_non_ascii_bytes(self):
        """Even with bytestrings, the mismatch should be coercible to unicode

        DocTestMatches is intended for text, but the Python 2 str type also
        permits arbitrary binary inputs. This is a slightly bogus thing to do,
        and under Python 3 using bytes objects will reasonably raise an error.
        """
        header = _b('\x89PNG\r\n\x1a\n...')
        self.assertRaises(TypeError, DocTestMatches, header, doctest.ELLIPSIS)