import io
import os
import shutil
import sys
import tempfile
from dulwich.tests import SkipTest, TestCase
from ..file import FileLocked, GitFile, _fancy_rename
class FancyRenameTests(TestCase):

    def setUp(self):
        super().setUp()
        self._tempdir = tempfile.mkdtemp()
        self.foo = self.path('foo')
        self.bar = self.path('bar')
        self.create(self.foo, b'foo contents')

    def tearDown(self):
        shutil.rmtree(self._tempdir)
        super().tearDown()

    def path(self, filename):
        return os.path.join(self._tempdir, filename)

    def create(self, path, contents):
        f = open(path, 'wb')
        f.write(contents)
        f.close()

    def test_no_dest_exists(self):
        self.assertFalse(os.path.exists(self.bar))
        _fancy_rename(self.foo, self.bar)
        self.assertFalse(os.path.exists(self.foo))
        new_f = open(self.bar, 'rb')
        self.assertEqual(b'foo contents', new_f.read())
        new_f.close()

    def test_dest_exists(self):
        self.create(self.bar, b'bar contents')
        _fancy_rename(self.foo, self.bar)
        self.assertFalse(os.path.exists(self.foo))
        new_f = open(self.bar, 'rb')
        self.assertEqual(b'foo contents', new_f.read())
        new_f.close()

    def test_dest_opened(self):
        if sys.platform != 'win32':
            raise SkipTest('platform allows overwriting open files')
        self.create(self.bar, b'bar contents')
        dest_f = open(self.bar, 'rb')
        self.assertRaises(OSError, _fancy_rename, self.foo, self.bar)
        dest_f.close()
        self.assertTrue(os.path.exists(self.path('foo')))
        new_f = open(self.foo, 'rb')
        self.assertEqual(b'foo contents', new_f.read())
        new_f.close()
        new_f = open(self.bar, 'rb')
        self.assertEqual(b'bar contents', new_f.read())
        new_f.close()