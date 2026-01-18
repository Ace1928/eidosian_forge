import os
from io import BytesIO
from .. import bedding, ignores
from . import TestCase, TestCaseInTempDir, TestCaseWithTransport
class TestUserIgnores(TestCaseInTempDir):

    def test_create_if_missing(self):
        ignore_path = bedding.user_ignore_config_path()
        self.assertPathDoesNotExist(ignore_path)
        user_ignores = ignores.get_user_ignores()
        self.assertEqual(set(ignores.USER_DEFAULTS), user_ignores)
        self.assertPathExists(ignore_path)
        with open(ignore_path, 'rb') as f:
            entries = ignores.parse_ignore_file(f)
        self.assertEqual(set(ignores.USER_DEFAULTS), entries)

    def test_create_with_intermediate_missing(self):
        ignore_path = bedding.user_ignore_config_path()
        self.assertPathDoesNotExist(ignore_path)
        os.mkdir('empty-home')
        config_path = os.path.join(self.test_dir, 'empty-home', 'foo', '.config')
        self.overrideEnv('BRZ_HOME', config_path)
        self.assertPathDoesNotExist(config_path)
        user_ignores = ignores.get_user_ignores()
        self.assertEqual(set(ignores.USER_DEFAULTS), user_ignores)
        ignore_path = bedding.user_ignore_config_path()
        self.assertPathDoesNotExist(ignore_path)

    def test_use_existing(self):
        patterns = ['*.o', '*.py[co]', 'å*']
        ignores._set_user_ignores(patterns)
        user_ignores = ignores.get_user_ignores()
        self.assertEqual(set(patterns), user_ignores)

    def test_use_empty(self):
        ignores._set_user_ignores([])
        ignore_path = bedding.user_ignore_config_path()
        self.check_file_contents(ignore_path, b'')
        self.assertEqual(set(), ignores.get_user_ignores())

    def test_set(self):
        patterns = ['*.py[co]', '*.py[oc]']
        ignores._set_user_ignores(patterns)
        self.assertEqual(set(patterns), ignores.get_user_ignores())
        patterns = ['vim', '*.swp']
        ignores._set_user_ignores(patterns)
        self.assertEqual(set(patterns), ignores.get_user_ignores())

    def test_add(self):
        """Test that adding will not duplicate ignores"""
        ignores._set_user_ignores([])
        patterns = ['foo', './bar', 'båz']
        added = ignores.add_unique_user_ignores(patterns)
        self.assertEqual(patterns, added)
        self.assertEqual(set(patterns), ignores.get_user_ignores())

    def test_add_directory(self):
        """Test that adding a directory will strip any trailing slash"""
        ignores._set_user_ignores([])
        in_patterns = ['foo/', 'bar/', 'baz\\']
        added = ignores.add_unique_user_ignores(in_patterns)
        out_patterns = [x.rstrip('/\\') for x in in_patterns]
        self.assertEqual(out_patterns, added)
        self.assertEqual(set(out_patterns), ignores.get_user_ignores())

    def test_add_unique(self):
        """Test that adding will not duplicate ignores"""
        ignores._set_user_ignores(['foo', './bar', 'båz', 'dir1/', 'dir3\\'])
        added = ignores.add_unique_user_ignores(['xxx', './bar', 'xxx', 'dir1/', 'dir2/', 'dir3\\'])
        self.assertEqual(['xxx', 'dir2'], added)
        self.assertEqual({'foo', './bar', 'båz', 'xxx', 'dir1', 'dir2', 'dir3'}, ignores.get_user_ignores())