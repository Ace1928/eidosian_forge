import errno
from .. import osutils, tests
from . import features
class TestWin32Finder(tests.TestCaseInTempDir):
    _test_needs_features = [win32_readdir_feature]

    def setUp(self):
        super().setUp()
        from ._walkdirs_win32 import Win32ReadDir
        self.reader = Win32ReadDir()

    def _remove_stat_from_dirblock(self, dirblock):
        return [info[:3] + info[4:] for info in dirblock]

    def assertWalkdirs(self, expected, top, prefix=''):
        old_selected_dir_reader = osutils._selected_dir_reader
        try:
            osutils._selected_dir_reader = self.reader
            finder = osutils._walkdirs_utf8(top, prefix=prefix)
            result = []
            for dirname, dirblock in finder:
                dirblock = self._remove_stat_from_dirblock(dirblock)
                result.append((dirname, dirblock))
            self.assertEqual(expected, result)
        finally:
            osutils._selected_dir_reader = old_selected_dir_reader

    def assertReadDir(self, expected, prefix, top_unicode):
        result = self._remove_stat_from_dirblock(self.reader.read_dir(prefix, top_unicode))
        self.assertEqual(expected, result)

    def test_top_prefix_to_starting_dir(self):
        self.assertEqual(('prefix', None, None, None, '\x12'), self.reader.top_prefix_to_starting_dir('\x12'.encode(), 'prefix'))

    def test_empty_directory(self):
        self.assertReadDir([], 'prefix', '.')
        self.assertWalkdirs([(('', '.'), [])], '.')

    def test_file(self):
        self.build_tree(['foo'])
        self.assertReadDir([('foo', 'foo', 'file', './foo')], '', '.')

    def test_directory(self):
        self.build_tree(['bar/'])
        self.assertReadDir([('bar', 'bar', 'directory', './bar')], '', '.')

    def test_prefix(self):
        self.build_tree(['bar/', 'baf'])
        self.assertReadDir([('xxx/baf', 'baf', 'file', './baf'), ('xxx/bar', 'bar', 'directory', './bar')], 'xxx', '.')

    def test_missing_dir(self):
        e = self.assertRaises(WindowsError, self.reader.read_dir, 'prefix', 'no_such_dir')
        self.assertEqual(errno.ENOENT, e.errno)
        self.assertEqual(3, e.winerror)
        self.assertEqual((3, 'no_such_dir/*'), e.args)