import os
import sys
from .. import bedding, osutils, tests
class TestConfigPath(tests.TestCase):

    def setUp(self):
        super().setUp()
        self.overrideEnv('HOME', '/home/bogus')
        self.overrideEnv('XDG_CACHE_HOME', '')
        if sys.platform == 'win32':
            self.overrideEnv('BRZ_HOME', 'C:\\Documents and Settings\\bogus\\Application Data')
            self.brz_home = 'C:/Documents and Settings/bogus/Application Data/breezy'
        else:
            self.brz_home = '/home/bogus/.config/breezy'

    def test_config_dir(self):
        self.assertEqual(bedding.config_dir(), self.brz_home)

    def test_config_dir_is_unicode(self):
        self.assertIsInstance(bedding.config_dir(), str)

    def test_config_path(self):
        self.assertEqual(bedding.config_path(), self.brz_home + '/breezy.conf')

    def test_locations_config_path(self):
        self.assertEqual(bedding.locations_config_path(), self.brz_home + '/locations.conf')

    def test_authentication_config_path(self):
        self.assertEqual(bedding.authentication_config_path(), self.brz_home + '/authentication.conf')