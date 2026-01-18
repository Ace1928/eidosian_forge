import bisect
import os
import time
from ... import osutils, tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios, multiply_scenarios
from ...tests.test_osutils import dir_reader_scenarios
from .. import _dirstate_helpers_py, dirstate
from . import test_dirstate
class TestLtByDirs(tests.TestCase):
    """Test an implementation of lt_by_dirs()

    lt_by_dirs() compares 2 paths by their directory sections, rather than as
    plain strings.

    Child test cases can override ``get_lt_by_dirs`` to test a specific
    implementation.
    """

    def get_lt_by_dirs(self):
        """Get a specific implementation of lt_by_dirs."""
        from .._dirstate_helpers_py import lt_by_dirs
        return lt_by_dirs

    def assertCmpByDirs(self, expected, str1, str2):
        """Compare the two strings, in both directions.

        :param expected: The expected comparison value. -1 means str1 comes
            first, 0 means they are equal, 1 means str2 comes first
        :param str1: string to compare
        :param str2: string to compare
        """
        lt_by_dirs = self.get_lt_by_dirs()
        if expected == 0:
            self.assertEqual(str1, str2)
            self.assertFalse(lt_by_dirs(str1, str2))
            self.assertFalse(lt_by_dirs(str2, str1))
        elif expected > 0:
            self.assertFalse(lt_by_dirs(str1, str2))
            self.assertTrue(lt_by_dirs(str2, str1))
        else:
            self.assertTrue(lt_by_dirs(str1, str2))
            self.assertFalse(lt_by_dirs(str2, str1))

    def test_cmp_empty(self):
        """Compare against the empty string."""
        self.assertCmpByDirs(0, b'', b'')
        self.assertCmpByDirs(1, b'a', b'')
        self.assertCmpByDirs(1, b'ab', b'')
        self.assertCmpByDirs(1, b'abc', b'')
        self.assertCmpByDirs(1, b'abcd', b'')
        self.assertCmpByDirs(1, b'abcde', b'')
        self.assertCmpByDirs(1, b'abcdef', b'')
        self.assertCmpByDirs(1, b'abcdefg', b'')
        self.assertCmpByDirs(1, b'abcdefgh', b'')
        self.assertCmpByDirs(1, b'abcdefghi', b'')
        self.assertCmpByDirs(1, b'test/ing/a/path/', b'')

    def test_cmp_same_str(self):
        """Compare the same string"""
        self.assertCmpByDirs(0, b'a', b'a')
        self.assertCmpByDirs(0, b'ab', b'ab')
        self.assertCmpByDirs(0, b'abc', b'abc')
        self.assertCmpByDirs(0, b'abcd', b'abcd')
        self.assertCmpByDirs(0, b'abcde', b'abcde')
        self.assertCmpByDirs(0, b'abcdef', b'abcdef')
        self.assertCmpByDirs(0, b'abcdefg', b'abcdefg')
        self.assertCmpByDirs(0, b'abcdefgh', b'abcdefgh')
        self.assertCmpByDirs(0, b'abcdefghi', b'abcdefghi')
        self.assertCmpByDirs(0, b'testing a long string', b'testing a long string')
        self.assertCmpByDirs(0, b'x' * 10000, b'x' * 10000)
        self.assertCmpByDirs(0, b'a/b', b'a/b')
        self.assertCmpByDirs(0, b'a/b/c', b'a/b/c')
        self.assertCmpByDirs(0, b'a/b/c/d', b'a/b/c/d')
        self.assertCmpByDirs(0, b'a/b/c/d/e', b'a/b/c/d/e')

    def test_simple_paths(self):
        """Compare strings that act like normal string comparison"""
        self.assertCmpByDirs(-1, b'a', b'b')
        self.assertCmpByDirs(-1, b'aa', b'ab')
        self.assertCmpByDirs(-1, b'ab', b'bb')
        self.assertCmpByDirs(-1, b'aaa', b'aab')
        self.assertCmpByDirs(-1, b'aab', b'abb')
        self.assertCmpByDirs(-1, b'abb', b'bbb')
        self.assertCmpByDirs(-1, b'aaaa', b'aaab')
        self.assertCmpByDirs(-1, b'aaab', b'aabb')
        self.assertCmpByDirs(-1, b'aabb', b'abbb')
        self.assertCmpByDirs(-1, b'abbb', b'bbbb')
        self.assertCmpByDirs(-1, b'aaaaa', b'aaaab')
        self.assertCmpByDirs(-1, b'a/a', b'a/b')
        self.assertCmpByDirs(-1, b'a/b', b'b/b')
        self.assertCmpByDirs(-1, b'a/a/a', b'a/a/b')
        self.assertCmpByDirs(-1, b'a/a/b', b'a/b/b')
        self.assertCmpByDirs(-1, b'a/b/b', b'b/b/b')
        self.assertCmpByDirs(-1, b'a/a/a/a', b'a/a/a/b')
        self.assertCmpByDirs(-1, b'a/a/a/b', b'a/a/b/b')
        self.assertCmpByDirs(-1, b'a/a/b/b', b'a/b/b/b')
        self.assertCmpByDirs(-1, b'a/b/b/b', b'b/b/b/b')
        self.assertCmpByDirs(-1, b'a/a/a/a/a', b'a/a/a/a/b')

    def test_tricky_paths(self):
        self.assertCmpByDirs(1, b'ab/cd/ef', b'ab/cc/ef')
        self.assertCmpByDirs(1, b'ab/cd/ef', b'ab/c/ef')
        self.assertCmpByDirs(-1, b'ab/cd/ef', b'ab/cd-ef')
        self.assertCmpByDirs(-1, b'ab/cd', b'ab/cd-')
        self.assertCmpByDirs(-1, b'ab/cd', b'ab-cd')

    def test_cmp_unicode_not_allowed(self):
        lt_by_dirs = self.get_lt_by_dirs()
        self.assertRaises(TypeError, lt_by_dirs, 'Unicode', b'str')
        self.assertRaises(TypeError, lt_by_dirs, b'str', 'Unicode')
        self.assertRaises(TypeError, lt_by_dirs, 'Unicode', 'Unicode')

    def test_cmp_non_ascii(self):
        self.assertCmpByDirs(-1, b'\xc2\xb5', b'\xc3\xa5')
        self.assertCmpByDirs(-1, b'a', b'\xc3\xa5')
        self.assertCmpByDirs(-1, b'b', b'\xc2\xb5')
        self.assertCmpByDirs(-1, b'a/b', b'a/\xc3\xa5')
        self.assertCmpByDirs(-1, b'b/a', b'b/\xc2\xb5')