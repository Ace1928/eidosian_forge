import os
import sys
from .. import bedding, osutils, tests
class TestConfigPathFallback(tests.TestCaseInTempDir):

    def setUp(self):
        super().setUp()
        self.overrideEnv('HOME', self.test_dir)
        self.overrideEnv('XDG_CACHE_HOME', '')
        self.bzr_home = os.path.join(self.test_dir, '.bazaar')
        os.mkdir(self.bzr_home)

    def test_config_dir(self):
        self.assertEqual(bedding.config_dir(), self.bzr_home)

    def test_config_dir_is_unicode(self):
        self.assertIsInstance(bedding.config_dir(), str)

    def test_config_path(self):
        self.assertEqual(bedding.config_path(), self.bzr_home + '/bazaar.conf')

    def test_locations_config_path(self):
        self.assertEqual(bedding.locations_config_path(), self.bzr_home + '/locations.conf')

    def test_authentication_config_path(self):
        self.assertEqual(bedding.authentication_config_path(), self.bzr_home + '/authentication.conf')