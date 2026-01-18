import sys
from breezy import rules, tests
class TestRulesPath(tests.TestCase):

    def setUp(self):
        super().setUp()
        self.overrideEnv('HOME', '/home/bogus')
        if sys.platform == 'win32':
            self.overrideEnv('BRZ_HOME', 'C:\\Documents and Settings\\bogus\\Application Data')
            self.brz_home = 'C:/Documents and Settings/bogus/Application Data/breezy'
        else:
            self.brz_home = '/home/bogus/.config/breezy'

    def test_rules_path(self):
        self.assertEqual(rules.rules_path(), self.brz_home + '/rules')