import io
from .. import errors, i18n, tests, workingtree
class TestGetText(tests.TestCase):

    def setUp(self):
        super().setUp()
        self.overrideAttr(i18n, '_translations', ZzzTranslations())

    def test_oneline(self):
        self.assertEqual('zzå{{spam ham eggs}}', i18n.gettext('spam ham eggs'))

    def test_multiline(self):
        self.assertEqual('zzå{{spam\nham\n\neggs\n}}', i18n.gettext('spam\nham\n\neggs\n'))