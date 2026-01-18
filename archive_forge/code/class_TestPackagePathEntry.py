import testtools
import fixtures
from fixtures import (
class TestPackagePathEntry(testtools.TestCase):

    def test_adds_missing_to_end_package_path(self):
        uniquedir = self.useFixture(TempDir()).path
        fixture = PackagePathEntry('fixtures', uniquedir)
        self.assertFalse(uniquedir in fixtures.__path__)
        with fixture:
            self.assertTrue(uniquedir in fixtures.__path__)
        self.assertFalse(uniquedir in fixtures.__path__)

    def test_doesnt_alter_existing_entry(self):
        existingdir = fixtures.__path__[0]
        expectedlen = len(fixtures.__path__)
        fixture = PackagePathEntry('fixtures', existingdir)
        with fixture:
            self.assertTrue(existingdir in fixtures.__path__)
            self.assertEqual(expectedlen, len(fixtures.__path__))
        self.assertTrue(existingdir in fixtures.__path__)
        self.assertEqual(expectedlen, len(fixtures.__path__))