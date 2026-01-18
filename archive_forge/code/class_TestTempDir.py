import os
import testtools
from testtools.matchers import StartsWith
from fixtures import (
class TestTempDir(testtools.TestCase):

    def test_basic(self):
        fixture = TempHomeDir()
        sentinel = object()
        self.assertEqual(sentinel, getattr(fixture, 'path', sentinel))
        fixture.setUp()
        try:
            path = fixture.path
            self.assertTrue(os.path.isdir(path))
            self.assertEqual(path, os.environ.get('HOME'))
        finally:
            fixture.cleanUp()
            self.assertFalse(os.path.isdir(path))

    def test_under_dir(self):
        root = self.useFixture(TempDir()).path
        fixture = TempHomeDir(root)
        fixture.setUp()
        with fixture:
            self.assertThat(fixture.path, StartsWith(root))