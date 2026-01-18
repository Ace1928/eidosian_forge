import errno
from .. import osutils, tests
from . import features
class Test_Win32Stat(tests.TestCaseInTempDir):
    _test_needs_features = [win32_readdir_feature]

    def setUp(self):
        super().setUp()
        from ._walkdirs_win32 import lstat
        self.win32_lstat = lstat

    def test_zero_members_present(self):
        self.build_tree(['foo'])
        st = self.win32_lstat('foo')
        self.assertEqual(0, st.st_dev)
        self.assertEqual(0, st.st_ino)
        self.assertEqual(0, st.st_uid)
        self.assertEqual(0, st.st_gid)