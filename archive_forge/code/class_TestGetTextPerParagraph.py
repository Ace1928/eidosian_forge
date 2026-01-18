import io
from .. import errors, i18n, tests, workingtree
class TestGetTextPerParagraph(tests.TestCase):

    def setUp(self):
        super().setUp()
        self.overrideAttr(i18n, '_translations', ZzzTranslations())

    def test_oneline(self):
        self.assertEqual('zzå{{spam ham eggs}}', i18n.gettext_per_paragraph('spam ham eggs'))

    def test_multiline(self):
        self.assertEqual('zzå{{spam\nham}}\n\nzzå{{eggs\n}}', i18n.gettext_per_paragraph('spam\nham\n\neggs\n'))