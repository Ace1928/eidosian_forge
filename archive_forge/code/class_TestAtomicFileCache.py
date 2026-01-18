import shutil
import sys
import tempfile
import unittest
import httplib2
from lazr.restfulclient._browser import AtomicFileCache, safename
class TestAtomicFileCache(TestFileCacheInterface):
    """Tests for ``AtomicFileCache``."""
    file_cache_factory = AtomicFileCache

    @staticmethod
    def prefix_safename(x):
        if isinstance(x, binary_type):
            x = x.decode('utf-8')
        return AtomicFileCache.TEMPFILE_PREFIX + x

    def test_set_non_string_value(self):
        cache = self.make_file_cache()
        self.assertRaises(TypeError, cache.set, 'answer', 42)
        self.assertIs(None, cache.get('answer'))

    def test_bad_safename_get(self):
        safename = self.prefix_safename
        cache = AtomicFileCache(self.cache_dir, safename)
        self.assertRaises(ValueError, cache.get, 'key')

    def test_bad_safename_set(self):
        safename = self.prefix_safename
        cache = AtomicFileCache(self.cache_dir, safename)
        self.assertRaises(ValueError, cache.set, 'key', b'value')

    def test_bad_safename_delete(self):
        safename = self.prefix_safename
        cache = AtomicFileCache(self.cache_dir, safename)
        self.assertRaises(ValueError, cache.delete, 'key')