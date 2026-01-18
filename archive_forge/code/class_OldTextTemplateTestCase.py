import doctest
import os
import shutil
import tempfile
import unittest
from genshi.template.base import TemplateSyntaxError
from genshi.template.loader import TemplateLoader
from genshi.template.text import OldTextTemplate, NewTextTemplate
class OldTextTemplateTestCase(unittest.TestCase):
    """Tests for text template processing."""

    def setUp(self):
        self.dirname = tempfile.mkdtemp(suffix='markup_test')

    def tearDown(self):
        shutil.rmtree(self.dirname)

    def test_escaping(self):
        tmpl = OldTextTemplate('\\#escaped')
        self.assertEqual('#escaped', tmpl.generate().render(encoding=None))

    def test_comment(self):
        tmpl = OldTextTemplate('## a comment')
        self.assertEqual('', tmpl.generate().render(encoding=None))

    def test_comment_escaping(self):
        tmpl = OldTextTemplate('\\## escaped comment')
        self.assertEqual('## escaped comment', tmpl.generate().render(encoding=None))

    def test_end_with_args(self):
        tmpl = OldTextTemplate("\n        #if foo\n          bar\n        #end 'if foo'")
        self.assertEqual('\n', tmpl.generate(foo=False).render(encoding=None))

    def test_latin1_encoded(self):
        text = u'$fooö$bar'.encode('iso-8859-1')
        tmpl = OldTextTemplate(text, encoding='iso-8859-1')
        self.assertEqual(u'xöy', tmpl.generate(foo='x', bar='y').render(encoding=None))

    def test_unicode_input(self):
        text = u'$fooö$bar'
        tmpl = OldTextTemplate(text)
        self.assertEqual(u'xöy', tmpl.generate(foo='x', bar='y').render(encoding=None))

    def test_empty_lines1(self):
        tmpl = OldTextTemplate('Your items:\n\n        #for item in items\n          * ${item}\n        #end')
        self.assertEqual('Your items:\n\n          * 0\n          * 1\n          * 2\n', tmpl.generate(items=range(3)).render(encoding=None))

    def test_empty_lines2(self):
        tmpl = OldTextTemplate('Your items:\n\n        #for item in items\n          * ${item}\n\n        #end')
        self.assertEqual('Your items:\n\n          * 0\n\n          * 1\n\n          * 2\n\n', tmpl.generate(items=range(3)).render(encoding=None))

    def test_include(self):
        file1 = open(os.path.join(self.dirname, 'tmpl1.txt'), 'wb')
        try:
            file1.write(u'Included\n'.encode('utf-8'))
        finally:
            file1.close()
        file2 = open(os.path.join(self.dirname, 'tmpl2.txt'), 'wb')
        try:
            file2.write(u'----- Included data below this line -----\n            #include tmpl1.txt\n            ----- Included data above this line -----'.encode('utf-8'))
        finally:
            file2.close()
        loader = TemplateLoader([self.dirname])
        tmpl = loader.load('tmpl2.txt', cls=OldTextTemplate)
        self.assertEqual('----- Included data below this line -----\nIncluded\n            ----- Included data above this line -----', tmpl.generate().render(encoding=None))