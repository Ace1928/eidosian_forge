import unittest
from genshi.core import Attrs, QName, Stream
from genshi.input import XMLParser, HTMLParser, ParseError, ET
from genshi.compat import StringIO, BytesIO
from genshi.tests.test_utils import doctest_suite
from xml.etree import ElementTree
class XMLParserTestCase(unittest.TestCase):

    def test_text_node_pos_single_line(self):
        text = '<elem>foo bar</elem>'
        events = list(XMLParser(StringIO(text)))
        kind, data, pos = events[1]
        self.assertEqual(Stream.TEXT, kind)
        self.assertEqual('foo bar', data)
        self.assertEqual((None, 1, 6), pos)

    def test_text_node_pos_multi_line(self):
        text = '<elem>foo\nbar</elem>'
        events = list(XMLParser(StringIO(text)))
        kind, data, pos = events[1]
        self.assertEqual(Stream.TEXT, kind)
        self.assertEqual('foo\nbar', data)
        self.assertEqual((None, 1, -1), pos)

    def test_element_attribute_order(self):
        text = '<elem title="baz" id="foo" class="bar" />'
        events = list(XMLParser(StringIO(text)))
        kind, data, pos = events[0]
        self.assertEqual(Stream.START, kind)
        tag, attrib = data
        self.assertEqual('elem', tag)
        self.assertEqual(('title', 'baz'), attrib[0])
        self.assertEqual(('id', 'foo'), attrib[1])
        self.assertEqual(('class', 'bar'), attrib[2])

    def test_unicode_input(self):
        text = u'<div>–</div>'
        events = list(XMLParser(StringIO(text)))
        kind, data, pos = events[1]
        self.assertEqual(Stream.TEXT, kind)
        self.assertEqual(u'–', data)

    def test_latin1_encoded(self):
        text = u'<div>ö</div>'.encode('iso-8859-1')
        events = list(XMLParser(BytesIO(text), encoding='iso-8859-1'))
        kind, data, pos = events[1]
        self.assertEqual(Stream.TEXT, kind)
        self.assertEqual(u'ö', data)

    def test_latin1_encoded_xmldecl(self):
        text = u'<?xml version="1.0" encoding="iso-8859-1" ?>\n        <div>ö</div>\n        '.encode('iso-8859-1')
        events = list(XMLParser(BytesIO(text)))
        kind, data, pos = events[2]
        self.assertEqual(Stream.TEXT, kind)
        self.assertEqual(u'ö', data)

    def test_html_entity_with_dtd(self):
        text = '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"\n        "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">\n        <html>&nbsp;</html>\n        '
        events = list(XMLParser(StringIO(text)))
        kind, data, pos = events[2]
        self.assertEqual(Stream.TEXT, kind)
        self.assertEqual(u'\xa0', data)

    def test_html_entity_without_dtd(self):
        text = '<html>&nbsp;</html>'
        events = list(XMLParser(StringIO(text)))
        kind, data, pos = events[1]
        self.assertEqual(Stream.TEXT, kind)
        self.assertEqual(u'\xa0', data)

    def test_html_entity_in_attribute(self):
        text = '<p title="&nbsp;"/>'
        events = list(XMLParser(StringIO(text)))
        kind, data, pos = events[0]
        self.assertEqual(Stream.START, kind)
        self.assertEqual(u'\xa0', data[1].get('title'))
        kind, data, pos = events[1]
        self.assertEqual(Stream.END, kind)

    def test_undefined_entity_with_dtd(self):
        text = '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"\n        "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">\n        <html>&junk;</html>\n        '
        events = XMLParser(StringIO(text))
        self.assertRaises(ParseError, list, events)

    def test_undefined_entity_without_dtd(self):
        text = '<html>&junk;</html>'
        events = XMLParser(StringIO(text))
        self.assertRaises(ParseError, list, events)