from breezy import config, i18n, tests
from breezy.tests.test_i18n import ZzzTranslations
class TestTranslatedHelp(tests.TestCaseWithTransport):
    """Tests for display of translated help topics"""

    def setUp(self):
        super().setUp()
        self.overrideAttr(i18n, '_translations', ZzzTranslations())

    def test_help_command_utf8(self):
        out, err = self.run_bzr_raw(['help', 'push'], encoding='utf-8')
        self.assertContainsRe(out, b'zz\xc3\xa5{{:See also:')

    def test_help_switch_utf8(self):
        out, err = self.run_bzr_raw(['push', '--help'], encoding='utf-8')
        self.assertContainsRe(out, b'zz\xc3\xa5{{:See also:')

    def test_help_command_ascii(self):
        out, err = self.run_bzr_raw(['help', 'push'], encoding='ascii')
        self.assertContainsRe(out, b'zz\\?{{:See also:')

    def test_help_switch_ascii(self):
        out, err = self.run_bzr_raw(['push', '--help'], encoding='ascii')
        self.assertContainsRe(out, b'zz\\?{{:See also:')