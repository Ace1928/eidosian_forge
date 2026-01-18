from ...tests import TestCase
from .classify import classify_filename
class TestClassify(TestCase):

    def test_classify_code(self):
        self.assertEqual('code', classify_filename('foo/bar.c'))
        self.assertEqual('code', classify_filename('foo/bar.pl'))
        self.assertEqual('code', classify_filename('foo/bar.pm'))

    def test_classify_documentation(self):
        self.assertEqual('documentation', classify_filename('bla.html'))

    def test_classify_translation(self):
        self.assertEqual('translation', classify_filename('nl.po'))

    def test_classify_art(self):
        self.assertEqual('art', classify_filename('icon.png'))

    def test_classify_unknown(self):
        self.assertEqual(None, classify_filename('something.bar'))

    def test_classify_doc_hardcoded(self):
        self.assertEqual('documentation', classify_filename('README'))

    def test_classify_multiple_periods(self):
        self.assertEqual('documentation', classify_filename('foo.bla.html'))