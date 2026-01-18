import unittest
from genshi.core import Attrs, Markup, QName, Stream
from genshi.input import HTML, XML
from genshi.output import DocType, XMLSerializer, XHTMLSerializer, \
from genshi.tests.test_utils import doctest_suite
class HTMLSerializerTestCase(unittest.TestCase):

    def test_xml_lang(self):
        text = '<p xml:lang="en">English text</p>'
        output = XML(text).render(HTMLSerializer, encoding=None)
        self.assertEqual('<p lang="en">English text</p>', output)

    def test_xml_lang_nodup(self):
        text = '<p lang="en" xml:lang="en">English text</p>'
        output = XML(text).render(HTMLSerializer, encoding=None)
        self.assertEqual('<p lang="en">English text</p>', output)

    def test_textarea_whitespace(self):
        content = '\nHey there.  \n\n    I am indented.\n'
        stream = XML('<textarea name="foo">%s</textarea>' % content)
        output = stream.render(HTMLSerializer, encoding=None)
        self.assertEqual('<textarea name="foo">%s</textarea>' % content, output)

    def test_pre_whitespace(self):
        content = '\nHey <em>there</em>.  \n\n    I am indented.\n'
        stream = XML('<pre>%s</pre>' % content)
        output = stream.render(HTMLSerializer, encoding=None)
        self.assertEqual('<pre>%s</pre>' % content, output)

    def test_xml_space(self):
        text = '<foo xml:space="preserve"> Do not mess  \n\n with me </foo>'
        output = XML(text).render(HTMLSerializer, encoding=None)
        self.assertEqual('<foo> Do not mess  \n\n with me </foo>', output)

    def test_empty_script(self):
        text = '<script src="foo.js" />'
        output = XML(text).render(HTMLSerializer, encoding=None)
        self.assertEqual('<script src="foo.js"></script>', output)

    def test_script_escaping(self):
        text = '<script>if (1 &lt; 2) { alert("Doh"); }</script>'
        output = XML(text).render(HTMLSerializer, encoding=None)
        self.assertEqual('<script>if (1 < 2) { alert("Doh"); }</script>', output)

    def test_script_escaping_with_namespace(self):
        text = '<script xmlns="http://www.w3.org/1999/xhtml">\n            if (1 &lt; 2) { alert("Doh"); }\n        </script>'
        output = XML(text).render(HTMLSerializer, encoding=None)
        self.assertEqual('<script>\n            if (1 < 2) { alert("Doh"); }\n        </script>', output)

    def test_style_escaping(self):
        text = '<style>html &gt; body { display: none; }</style>'
        output = XML(text).render(HTMLSerializer, encoding=None)
        self.assertEqual('<style>html > body { display: none; }</style>', output)

    def test_style_escaping_with_namespace(self):
        text = '<style xmlns="http://www.w3.org/1999/xhtml">\n            html &gt; body { display: none; }\n        </style>'
        output = XML(text).render(HTMLSerializer, encoding=None)
        self.assertEqual('<style>\n            html > body { display: none; }\n        </style>', output)

    def test_html5_doctype(self):
        stream = HTML(u'<html></html>')
        output = stream.render(HTMLSerializer, doctype=DocType.HTML5, encoding=None)
        self.assertEqual('<!DOCTYPE html>\n<html></html>', output)