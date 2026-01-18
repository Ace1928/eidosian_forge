import io
from .. import errors, i18n, tests, workingtree
class LoadPluginTranslations(tests.TestCase):

    def test_does_not_exist(self):
        translation = i18n.load_plugin_translations('doesnotexist')
        self.assertEqual('foo', translation.gettext('foo'))