import unittest
import tornado
from tornado.escape import (
from tornado.util import unicode_type
from typing import List, Tuple, Union, Dict, Any  # noqa: F401
class EscapeTestCase(unittest.TestCase):

    def test_linkify(self):
        for text, kwargs, html in linkify_tests:
            linked = tornado.escape.linkify(text, **kwargs)
            self.assertEqual(linked, html)

    def test_xhtml_escape(self):
        tests = [('<foo>', '&lt;foo&gt;'), ('<foo>', '&lt;foo&gt;'), (b'<foo>', b'&lt;foo&gt;'), ('<>&"\'', '&lt;&gt;&amp;&quot;&#x27;'), ('&amp;', '&amp;amp;'), ('<é>', '&lt;é&gt;'), (b'<\xc3\xa9>', b'&lt;\xc3\xa9&gt;')]
        for unescaped, escaped in tests:
            self.assertEqual(utf8(xhtml_escape(unescaped)), utf8(escaped))
            self.assertEqual(utf8(unescaped), utf8(xhtml_unescape(escaped)))

    def test_xhtml_unescape_numeric(self):
        tests = [('foo&#32;bar', 'foo bar'), ('foo&#x20;bar', 'foo bar'), ('foo&#X20;bar', 'foo bar'), ('foo&#xabc;bar', 'foo઼bar'), ('foo&#xyz;bar', 'foo&#xyz;bar'), ('foo&#;bar', 'foo&#;bar'), ('foo&#x;bar', 'foo&#x;bar')]
        for escaped, unescaped in tests:
            self.assertEqual(unescaped, xhtml_unescape(escaped))

    def test_url_escape_unicode(self):
        tests = [('é'.encode('utf8'), '%C3%A9'), ('é'.encode('latin1'), '%E9'), ('é', '%C3%A9')]
        for unescaped, escaped in tests:
            self.assertEqual(url_escape(unescaped), escaped)

    def test_url_unescape_unicode(self):
        tests = [('%C3%A9', 'é', 'utf8'), ('%C3%A9', 'Ã©', 'latin1'), ('%C3%A9', utf8('é'), None)]
        for escaped, unescaped, encoding in tests:
            self.assertEqual(url_unescape(to_unicode(escaped), encoding), unescaped)
            self.assertEqual(url_unescape(utf8(escaped), encoding), unescaped)

    def test_url_escape_quote_plus(self):
        unescaped = '+ #%'
        plus_escaped = '%2B+%23%25'
        escaped = '%2B%20%23%25'
        self.assertEqual(url_escape(unescaped), plus_escaped)
        self.assertEqual(url_escape(unescaped, plus=False), escaped)
        self.assertEqual(url_unescape(plus_escaped), unescaped)
        self.assertEqual(url_unescape(escaped, plus=False), unescaped)
        self.assertEqual(url_unescape(plus_escaped, encoding=None), utf8(unescaped))
        self.assertEqual(url_unescape(escaped, encoding=None, plus=False), utf8(unescaped))

    def test_escape_return_types(self):
        self.assertEqual(type(xhtml_escape('foo')), str)
        self.assertEqual(type(xhtml_escape('foo')), unicode_type)

    def test_json_decode(self):
        self.assertEqual(json_decode(b'"foo"'), 'foo')
        self.assertEqual(json_decode('"foo"'), 'foo')
        self.assertEqual(json_decode(utf8('"é"')), 'é')

    def test_json_encode(self):
        self.assertEqual(json_decode(json_encode('é')), 'é')
        if bytes is str:
            self.assertEqual(json_decode(json_encode(utf8('é'))), 'é')
            self.assertRaises(UnicodeDecodeError, json_encode, b'\xe9')

    def test_squeeze(self):
        self.assertEqual(squeeze('sequences     of    whitespace   chars'), 'sequences of whitespace chars')

    def test_recursive_unicode(self):
        tests = {'dict': {b'foo': b'bar'}, 'list': [b'foo', b'bar'], 'tuple': (b'foo', b'bar'), 'bytes': b'foo'}
        self.assertEqual(recursive_unicode(tests['dict']), {'foo': 'bar'})
        self.assertEqual(recursive_unicode(tests['list']), ['foo', 'bar'])
        self.assertEqual(recursive_unicode(tests['tuple']), ('foo', 'bar'))
        self.assertEqual(recursive_unicode(tests['bytes']), 'foo')