import datetime
import os
import shutil
import tempfile
import unittest
import tornado.locale
from tornado.escape import utf8, to_unicode
from tornado.util import unicode_type
class TranslationLoaderTest(unittest.TestCase):
    SAVE_VARS = ['_translations', '_supported_locales', '_use_gettext']

    def clear_locale_cache(self):
        tornado.locale.Locale._cache = {}

    def setUp(self):
        self.saved = {}
        for var in TranslationLoaderTest.SAVE_VARS:
            self.saved[var] = getattr(tornado.locale, var)
        self.clear_locale_cache()

    def tearDown(self):
        for k, v in self.saved.items():
            setattr(tornado.locale, k, v)
        self.clear_locale_cache()

    def test_csv(self):
        tornado.locale.load_translations(os.path.join(os.path.dirname(__file__), 'csv_translations'))
        locale = tornado.locale.get('fr_FR')
        self.assertTrue(isinstance(locale, tornado.locale.CSVLocale))
        self.assertEqual(locale.translate('school'), 'école')

    def test_csv_bom(self):
        with open(os.path.join(os.path.dirname(__file__), 'csv_translations', 'fr_FR.csv'), 'rb') as f:
            char_data = to_unicode(f.read())
        for encoding in ['utf-8-sig', 'utf-16']:
            tmpdir = tempfile.mkdtemp()
            try:
                with open(os.path.join(tmpdir, 'fr_FR.csv'), 'wb') as f:
                    f.write(char_data.encode(encoding))
                tornado.locale.load_translations(tmpdir)
                locale = tornado.locale.get('fr_FR')
                self.assertIsInstance(locale, tornado.locale.CSVLocale)
                self.assertEqual(locale.translate('school'), 'école')
            finally:
                shutil.rmtree(tmpdir)

    def test_gettext(self):
        tornado.locale.load_gettext_translations(os.path.join(os.path.dirname(__file__), 'gettext_translations'), 'tornado_test')
        locale = tornado.locale.get('fr_FR')
        self.assertTrue(isinstance(locale, tornado.locale.GettextLocale))
        self.assertEqual(locale.translate('school'), 'école')
        self.assertEqual(locale.pgettext('law', 'right'), 'le droit')
        self.assertEqual(locale.pgettext('good', 'right'), 'le bien')
        self.assertEqual(locale.pgettext('organization', 'club', 'clubs', 1), 'le club')
        self.assertEqual(locale.pgettext('organization', 'club', 'clubs', 2), 'les clubs')
        self.assertEqual(locale.pgettext('stick', 'club', 'clubs', 1), 'le bâton')
        self.assertEqual(locale.pgettext('stick', 'club', 'clubs', 2), 'les bâtons')