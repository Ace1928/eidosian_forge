import os
import stat
import sys
from .. import atomicfile, osutils
from . import TestCaseInTempDir, TestSkipped
class TestAtomicFile(TestCaseInTempDir):

    def test_commit(self):
        f = atomicfile.AtomicFile('test')
        self.assertPathDoesNotExist('test')
        f.write(b'foo\n')
        f.commit()
        self.assertEqual(['test'], os.listdir('.'))
        self.check_file_contents('test', b'foo\n')
        self.assertRaises(atomicfile.AtomicFileAlreadyClosed, f.commit)
        self.assertRaises(atomicfile.AtomicFileAlreadyClosed, f.abort)
        f.close()

    def test_abort(self):
        f = atomicfile.AtomicFile('test')
        f.write(b'foo\n')
        f.abort()
        self.assertEqual([], os.listdir('.'))
        self.assertRaises(atomicfile.AtomicFileAlreadyClosed, f.abort)
        self.assertRaises(atomicfile.AtomicFileAlreadyClosed, f.commit)
        f.close()

    def test_close(self):
        f = atomicfile.AtomicFile('test')
        f.write(b'foo\n')
        f.close()
        self.assertEqual([], os.listdir('.'))
        self.assertRaises(atomicfile.AtomicFileAlreadyClosed, f.abort)
        self.assertRaises(atomicfile.AtomicFileAlreadyClosed, f.commit)
        f.close()

    def test_text_mode(self):
        f = atomicfile.AtomicFile('test', mode='wt')
        f.write(b'foo\n')
        f.commit()
        with open('test', 'rb') as f:
            contents = f.read()
        if sys.platform == 'win32':
            self.assertEqual(b'foo\r\n', contents)
        else:
            self.assertEqual(b'foo\n', contents)

    def can_sys_preserve_mode(self):
        return sys.platform not in ('win32',)

    def _test_mode(self, mode):
        if not self.can_sys_preserve_mode():
            raise TestSkipped('This test cannot be run on your platform')
        f = atomicfile.AtomicFile('test', mode='wb', new_mode=mode)
        f.write(b'foo\n')
        f.commit()
        st = os.lstat('test')
        self.assertEqualMode(mode, stat.S_IMODE(st.st_mode))

    def test_mode_0666(self):
        self._test_mode(438)

    def test_mode_0664(self):
        self._test_mode(436)

    def test_mode_0660(self):
        self._test_mode(432)

    def test_mode_0640(self):
        self._test_mode(416)

    def test_mode_0600(self):
        self._test_mode(384)

    def test_mode_0400(self):
        self._test_mode(256)
        os.chmod('test', 384)

    def test_no_mode(self):
        umask = osutils.get_umask()
        f = atomicfile.AtomicFile('test', mode='wb')
        f.write(b'foo\n')
        f.commit()
        st = os.lstat('test')
        self.assertEqualMode(438 & ~umask, stat.S_IMODE(st.st_mode))

    def test_context_manager_commit(self):
        with atomicfile.AtomicFile('test') as f:
            self.assertPathDoesNotExist('test')
            f.write(b'foo\n')
        self.assertEqual(['test'], os.listdir('.'))
        self.check_file_contents('test', b'foo\n')

    def test_context_manager_abort(self):

        def abort():
            with atomicfile.AtomicFile('test') as f:
                f.write(b'foo\n')
                raise AssertionError
        self.assertRaises(AssertionError, abort)
        self.assertEqual([], os.listdir('.'))