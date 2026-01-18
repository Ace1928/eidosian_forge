import os
import tempfile
from fire import __main__
from fire import testutils
class MainModuleFileTest(testutils.BaseTestCase):
    """Tests to verify correct import behavior for file executables."""

    def setUp(self):
        super(MainModuleFileTest, self).setUp()
        self.file = tempfile.NamedTemporaryFile(suffix='.py')
        self.file.write(b'class Foo:\n  def double(self, n):\n    return 2 * n\n')
        self.file.flush()
        self.file2 = tempfile.NamedTemporaryFile()

    def testFileNameFire(self):
        with self.assertOutputMatches('4'):
            __main__.main(['__main__.py', self.file.name, 'Foo', 'double', '--n', '2'])

    def testFileNameFailure(self):
        with self.assertRaises(ValueError):
            __main__.main(['__main__.py', self.file2.name, 'Foo', 'double', '--n', '2'])

    def testFileNameModuleDuplication(self):
        with self.assertOutputMatches('gettempdir'):
            dirname = os.path.dirname(self.file.name)
            with testutils.ChangeDirectory(dirname):
                with open('tempfile', 'w'):
                    __main__.main(['__main__.py', 'tempfile'])
                os.remove('tempfile')

    def testFileNameModuleFileFailure(self):
        with self.assertRaisesRegex(ValueError, 'Fire can only be called on \\.py files\\.'):
            dirname = os.path.dirname(self.file.name)
            with testutils.ChangeDirectory(dirname):
                with open('foobar', 'w'):
                    __main__.main(['__main__.py', 'foobar'])
                os.remove('foobar')