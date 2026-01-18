import bisect
import os
import time
from ... import osutils, tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios, multiply_scenarios
from ...tests.test_osutils import dir_reader_scenarios
from .. import _dirstate_helpers_py, dirstate
from . import test_dirstate
class TestBisectDirblock(tests.TestCase):
    """Test that bisect_dirblock() returns the expected values.

    bisect_dirblock is intended to work like bisect.bisect_left() except it
    knows it is working on dirblocks and that dirblocks are sorted by ('path',
    'to', 'foo') chunks rather than by raw 'path/to/foo'.

    This test is parameterized by calling get_bisect_dirblock(). Child test
    cases can override this function to test against a different
    implementation.
    """

    def get_bisect_dirblock(self):
        """Return an implementation of bisect_dirblock"""
        from .._dirstate_helpers_py import bisect_dirblock
        return bisect_dirblock

    def assertBisect(self, dirblocks, split_dirblocks, path, *args, **kwargs):
        """Assert that bisect_split works like bisect_left on the split paths.

        :param dirblocks: A list of (path, [info]) pairs.
        :param split_dirblocks: A list of ((split, path), [info]) pairs.
        :param path: The path we are indexing.

        All other arguments will be passed along.
        """
        bisect_dirblock = self.get_bisect_dirblock()
        self.assertIsInstance(dirblocks, list)
        bisect_split_idx = bisect_dirblock(dirblocks, path, *args, **kwargs)
        split_dirblock = (path.split(b'/'), [])
        bisect_left_idx = bisect.bisect_left(split_dirblocks, split_dirblock, *args)
        self.assertEqual(bisect_left_idx, bisect_split_idx, 'bisect_split disagreed. %s != %s for key %r' % (bisect_left_idx, bisect_split_idx, path))

    def paths_to_dirblocks(self, paths):
        """Convert a list of paths into dirblock form.

        Also, ensure that the paths are in proper sorted order.
        """
        dirblocks = [(path, []) for path in paths]
        split_dirblocks = [(path.split(b'/'), []) for path in paths]
        self.assertEqual(sorted(split_dirblocks), split_dirblocks)
        return (dirblocks, split_dirblocks)

    def test_simple(self):
        """In the simple case it works just like bisect_left"""
        paths = [b'', b'a', b'b', b'c', b'd']
        dirblocks, split_dirblocks = self.paths_to_dirblocks(paths)
        for path in paths:
            self.assertBisect(dirblocks, split_dirblocks, path)
        self.assertBisect(dirblocks, split_dirblocks, b'_')
        self.assertBisect(dirblocks, split_dirblocks, b'aa')
        self.assertBisect(dirblocks, split_dirblocks, b'bb')
        self.assertBisect(dirblocks, split_dirblocks, b'cc')
        self.assertBisect(dirblocks, split_dirblocks, b'dd')
        self.assertBisect(dirblocks, split_dirblocks, b'a/a')
        self.assertBisect(dirblocks, split_dirblocks, b'b/b')
        self.assertBisect(dirblocks, split_dirblocks, b'c/c')
        self.assertBisect(dirblocks, split_dirblocks, b'd/d')

    def test_involved(self):
        """This is where bisect_left diverges slightly."""
        paths = [b'', b'a', b'a/a', b'a/a/a', b'a/a/z', b'a/a-a', b'a/a-z', b'a/z', b'a/z/a', b'a/z/z', b'a/z-a', b'a/z-z', b'a-a', b'a-z', b'z', b'z/a/a', b'z/a/z', b'z/a-a', b'z/a-z', b'z/z', b'z/z/a', b'z/z/z', b'z/z-a', b'z/z-z', b'z-a', b'z-z']
        dirblocks, split_dirblocks = self.paths_to_dirblocks(paths)
        for path in paths:
            self.assertBisect(dirblocks, split_dirblocks, path)

    def test_involved_cached(self):
        """This is where bisect_left diverges slightly."""
        paths = [b'', b'a', b'a/a', b'a/a/a', b'a/a/z', b'a/a-a', b'a/a-z', b'a/z', b'a/z/a', b'a/z/z', b'a/z-a', b'a/z-z', b'a-a', b'a-z', b'z', b'z/a/a', b'z/a/z', b'z/a-a', b'z/a-z', b'z/z', b'z/z/a', b'z/z/z', b'z/z-a', b'z/z-z', b'z-a', b'z-z']
        cache = {}
        dirblocks, split_dirblocks = self.paths_to_dirblocks(paths)
        for path in paths:
            self.assertBisect(dirblocks, split_dirblocks, path, cache=cache)