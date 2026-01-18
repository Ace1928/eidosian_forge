import bisect
import os
import time
from ... import osutils, tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios, multiply_scenarios
from ...tests.test_osutils import dir_reader_scenarios
from .. import _dirstate_helpers_py, dirstate
from . import test_dirstate
class TestBisectPathMixin:
    """Test that _bisect_path_*() returns the expected values.

    _bisect_path_* is intended to work like bisect.bisect_*() except it
    knows it is working on paths that are sorted by ('path', 'to', 'foo')
    chunks rather than by raw 'path/to/foo'.

    Test Cases should inherit from this and override ``get_bisect_path`` return
    their implementation, and ``get_bisect`` to return the matching
    bisect.bisect_* function.
    """

    def get_bisect_path(self):
        """Return an implementation of _bisect_path_*"""
        raise NotImplementedError

    def get_bisect(self):
        """Return a version of bisect.bisect_*.

        Also, for the 'exists' check, return the offset to the real values.
        For example bisect_left returns the index of an entry, while
        bisect_right returns the index *after* an entry

        :return: (bisect_func, offset)
        """
        raise NotImplementedError

    def assertBisect(self, paths, split_paths, path, exists=True):
        """Assert that bisect_split works like bisect_left on the split paths.

        :param paths: A list of path names
        :param split_paths: A list of path names that are already split up by directory
            ('path/to/foo' => ('path', 'to', 'foo'))
        :param path: The path we are indexing.
        :param exists: The path should be present, so make sure the
            final location actually points to the right value.

        All other arguments will be passed along.
        """
        bisect_path = self.get_bisect_path()
        self.assertIsInstance(paths, list)
        bisect_path_idx = bisect_path(paths, path)
        split_path = self.split_for_dirblocks([path])[0]
        bisect_func, offset = self.get_bisect()
        bisect_split_idx = bisect_func(split_paths, split_path)
        self.assertEqual(bisect_split_idx, bisect_path_idx, '%s disagreed. %s != %s for key %r' % (bisect_path.__name__, bisect_split_idx, bisect_path_idx, path))
        if exists:
            self.assertEqual(path, paths[bisect_path_idx + offset])

    def split_for_dirblocks(self, paths):
        dir_split_paths = []
        for path in paths:
            dirname, basename = os.path.split(path)
            dir_split_paths.append((dirname.split(b'/'), basename))
        dir_split_paths.sort()
        return dir_split_paths

    def test_simple(self):
        """In the simple case it works just like bisect_left"""
        paths = [b'', b'a', b'b', b'c', b'd']
        split_paths = self.split_for_dirblocks(paths)
        for path in paths:
            self.assertBisect(paths, split_paths, path, exists=True)
        self.assertBisect(paths, split_paths, b'_', exists=False)
        self.assertBisect(paths, split_paths, b'aa', exists=False)
        self.assertBisect(paths, split_paths, b'bb', exists=False)
        self.assertBisect(paths, split_paths, b'cc', exists=False)
        self.assertBisect(paths, split_paths, b'dd', exists=False)
        self.assertBisect(paths, split_paths, b'a/a', exists=False)
        self.assertBisect(paths, split_paths, b'b/b', exists=False)
        self.assertBisect(paths, split_paths, b'c/c', exists=False)
        self.assertBisect(paths, split_paths, b'd/d', exists=False)

    def test_involved(self):
        """This is where bisect_path_* diverges slightly."""
        paths = [b'', b'a', b'a-a', b'a-z', b'a=a', b'a=z', b'a/a', b'a/a-a', b'a/a-z', b'a/a=a', b'a/a=z', b'a/z', b'a/z-a', b'a/z-z', b'a/z=a', b'a/z=z', b'a/a/a', b'a/a/z', b'a/a-a/a', b'a/a-z/z', b'a/a=a/a', b'a/a=z/z', b'a/z/a', b'a/z/z', b'a-a/a', b'a-z/z', b'a=a/a', b'a=z/z']
        split_paths = self.split_for_dirblocks(paths)
        sorted_paths = []
        for dir_parts, basename in split_paths:
            if dir_parts == [b'']:
                sorted_paths.append(basename)
            else:
                sorted_paths.append(b'/'.join(dir_parts + [basename]))
        self.assertEqual(sorted_paths, paths)
        for path in paths:
            self.assertBisect(paths, split_paths, path, exists=True)