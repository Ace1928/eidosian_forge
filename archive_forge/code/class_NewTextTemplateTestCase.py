import doctest
import os
import shutil
import tempfile
import unittest
from genshi.template.base import TemplateSyntaxError
from genshi.template.loader import TemplateLoader
from genshi.template.text import OldTextTemplate, NewTextTemplate
class NewTextTemplateTestCase(unittest.TestCase):
    """Tests for text template processing."""

    def setUp(self):
        self.dirname = tempfile.mkdtemp(suffix='markup_test')

    def tearDown(self):
        shutil.rmtree(self.dirname)

    def test_escaping(self):
        tmpl = NewTextTemplate('\\{% escaped %}')
        self.assertEqual('{% escaped %}', tmpl.generate().render(encoding=None))

    def test_comment(self):
        tmpl = NewTextTemplate('{# a comment #}')
        self.assertEqual('', tmpl.generate().render(encoding=None))

    def test_comment_escaping(self):
        tmpl = NewTextTemplate('\\{# escaped comment #}')
        self.assertEqual('{# escaped comment #}', tmpl.generate().render(encoding=None))

    def test_end_with_args(self):
        tmpl = NewTextTemplate("\n{% if foo %}\n  bar\n{% end 'if foo' %}")
        self.assertEqual('\n', tmpl.generate(foo=False).render(encoding=None))

    def test_latin1_encoded(self):
        text = u'$fooö$bar'.encode('iso-8859-1')
        tmpl = NewTextTemplate(text, encoding='iso-8859-1')
        self.assertEqual(u'xöy', tmpl.generate(foo='x', bar='y').render(encoding=None))

    def test_unicode_input(self):
        text = u'$fooö$bar'
        tmpl = NewTextTemplate(text)
        self.assertEqual(u'xöy', tmpl.generate(foo='x', bar='y').render(encoding=None))

    def test_empty_lines1(self):
        tmpl = NewTextTemplate('Your items:\n\n{% for item in items %}  * ${item}\n{% end %}')
        self.assertEqual('Your items:\n\n  * 0\n  * 1\n  * 2\n', tmpl.generate(items=range(3)).render(encoding=None))

    def test_empty_lines1_with_crlf(self):
        tmpl = NewTextTemplate('Your items:\r\n\r\n{% for item in items %}\\\r\n  * ${item}\r\n{% end %}')
        self.assertEqual('Your items:\r\n\r\n  * 0\r\n  * 1\r\n  * 2\r\n', tmpl.generate(items=range(3)).render(encoding=None))

    def test_empty_lines2(self):
        tmpl = NewTextTemplate('Your items:\n\n{% for item in items %}  * ${item}\n\n{% end %}')
        self.assertEqual('Your items:\n\n  * 0\n\n  * 1\n\n  * 2\n\n', tmpl.generate(items=range(3)).render(encoding=None))

    def test_empty_lines2_with_crlf(self):
        tmpl = NewTextTemplate('Your items:\r\n\r\n{% for item in items %}\\\r\n  * ${item}\r\n\r\n{% end %}')
        self.assertEqual('Your items:\r\n\r\n  * 0\r\n\r\n  * 1\r\n\r\n  * 2\r\n\r\n', tmpl.generate(items=range(3)).render(encoding=None))

    def test_exec_with_trailing_space(self):
        """
        Verify that a code block with trailing space does not cause a syntax
        error (see ticket #127).
        """
        NewTextTemplate('\n          {% python\n            bar = 42\n          $}\n        ')

    def test_exec_import(self):
        tmpl = NewTextTemplate('{% python from datetime import timedelta %}\n        ${timedelta(days=2)}\n        ')
        self.assertEqual('\n        2 days, 0:00:00\n        ', tmpl.generate().render(encoding=None))

    def test_exec_def(self):
        tmpl = NewTextTemplate('{% python\n        def foo():\n            return 42\n        %}\n        ${foo()}\n        ')
        self.assertEqual('\n        42\n        ', tmpl.generate().render(encoding=None))

    def test_include(self):
        file1 = open(os.path.join(self.dirname, 'tmpl1.txt'), 'wb')
        try:
            file1.write(u'Included'.encode('utf-8'))
        finally:
            file1.close()
        file2 = open(os.path.join(self.dirname, 'tmpl2.txt'), 'wb')
        try:
            file2.write(u'----- Included data below this line -----\n{% include tmpl1.txt %}\n----- Included data above this line -----'.encode('utf-8'))
        finally:
            file2.close()
        loader = TemplateLoader([self.dirname])
        tmpl = loader.load('tmpl2.txt', cls=NewTextTemplate)
        self.assertEqual('----- Included data below this line -----\nIncluded\n----- Included data above this line -----', tmpl.generate().render(encoding=None))

    def test_include_expr(self):
        file1 = open(os.path.join(self.dirname, 'tmpl1.txt'), 'wb')
        try:
            file1.write(u'Included'.encode('utf-8'))
        finally:
            file1.close()
        file2 = open(os.path.join(self.dirname, 'tmpl2.txt'), 'wb')
        try:
            file2.write(u"----- Included data below this line -----\n    {% include ${'%s.txt' % ('tmpl1',)} %}\n    ----- Included data above this line -----".encode('utf-8'))
        finally:
            file2.close()
        loader = TemplateLoader([self.dirname])
        tmpl = loader.load('tmpl2.txt', cls=NewTextTemplate)
        self.assertEqual('----- Included data below this line -----\n    Included\n    ----- Included data above this line -----', tmpl.generate().render(encoding=None))