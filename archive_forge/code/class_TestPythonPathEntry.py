import sys
import testtools
from fixtures import (
class TestPythonPathEntry(testtools.TestCase):

    def test_adds_missing_to_end_sys_path(self):
        uniquedir = self.useFixture(TempDir()).path
        fixture = PythonPathEntry(uniquedir)
        self.assertFalse(uniquedir in sys.path)
        with fixture:
            self.assertTrue(uniquedir in sys.path)
        self.assertFalse(uniquedir in sys.path)

    def test_doesnt_alter_existing_entry(self):
        existingdir = sys.path[0]
        expectedlen = len(sys.path)
        fixture = PythonPathEntry(existingdir)
        with fixture:
            self.assertTrue(existingdir in sys.path)
            self.assertEqual(expectedlen, len(sys.path))
        self.assertTrue(existingdir in sys.path)
        self.assertEqual(expectedlen, len(sys.path))