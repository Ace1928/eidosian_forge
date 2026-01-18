from tornado.httputil import (
from tornado.escape import utf8, native_str
from tornado.log import gen_log
from tornado.testing import ExpectLog
from tornado.test.util import ignore_deprecation
import copy
import datetime
import logging
import pickle
import time
import urllib.parse
import unittest
from typing import Tuple, Dict, List
class HTTPHeadersTest(unittest.TestCase):

    def test_multi_line(self):
        data = 'Foo: bar\n baz\nAsdf: qwer\n\tzxcv\nFoo: even\n     more\n     lines\n'.replace('\n', '\r\n')
        headers = HTTPHeaders.parse(data)
        self.assertEqual(headers['asdf'], 'qwer zxcv')
        self.assertEqual(headers.get_list('asdf'), ['qwer zxcv'])
        self.assertEqual(headers['Foo'], 'bar baz,even more lines')
        self.assertEqual(headers.get_list('foo'), ['bar baz', 'even more lines'])
        self.assertEqual(sorted(list(headers.get_all())), [('Asdf', 'qwer zxcv'), ('Foo', 'bar baz'), ('Foo', 'even more lines')])

    def test_malformed_continuation(self):
        data = ' Foo: bar'
        self.assertRaises(HTTPInputError, HTTPHeaders.parse, data)

    def test_unicode_newlines(self):
        newlines = ['\x1b', '\x1c', '\x1d', '\x1e', '\x85', '\u2028', '\u2029']
        for newline in newlines:
            for encoding in ['utf8', 'latin1']:
                try:
                    try:
                        encoded = newline.encode(encoding)
                    except UnicodeEncodeError:
                        continue
                    data = b'Cookie: foo=' + encoded + b'bar'
                    headers = HTTPHeaders.parse(native_str(data.decode('latin1')))
                    expected = [('Cookie', 'foo=' + native_str(encoded.decode('latin1')) + 'bar')]
                    self.assertEqual(expected, list(headers.get_all()))
                except Exception:
                    gen_log.warning('failed while trying %r in %s', newline, encoding)
                    raise

    def test_optional_cr(self):
        headers = HTTPHeaders.parse('CRLF: crlf\r\nLF: lf\nCR: cr\rMore: more\r\n')
        self.assertEqual(sorted(headers.get_all()), [('Cr', 'cr\rMore: more'), ('Crlf', 'crlf'), ('Lf', 'lf')])

    def test_copy(self):
        all_pairs = [('A', '1'), ('A', '2'), ('B', 'c')]
        h1 = HTTPHeaders()
        for k, v in all_pairs:
            h1.add(k, v)
        h2 = h1.copy()
        h3 = copy.copy(h1)
        h4 = copy.deepcopy(h1)
        for headers in [h1, h2, h3, h4]:
            self.assertEqual(list(sorted(headers.get_all())), all_pairs)
        for headers in [h2, h3, h4]:
            self.assertIsNot(headers, h1)
            self.assertIsNot(headers.get_list('A'), h1.get_list('A'))

    def test_pickle_roundtrip(self):
        headers = HTTPHeaders()
        headers.add('Set-Cookie', 'a=b')
        headers.add('Set-Cookie', 'c=d')
        headers.add('Content-Type', 'text/html')
        pickled = pickle.dumps(headers)
        unpickled = pickle.loads(pickled)
        self.assertEqual(sorted(headers.get_all()), sorted(unpickled.get_all()))
        self.assertEqual(sorted(headers.items()), sorted(unpickled.items()))

    def test_setdefault(self):
        headers = HTTPHeaders()
        headers['foo'] = 'bar'
        self.assertEqual(headers.setdefault('foo', 'baz'), 'bar')
        self.assertEqual(headers['foo'], 'bar')
        self.assertEqual(headers.setdefault('quux', 'xyzzy'), 'xyzzy')
        self.assertEqual(headers['quux'], 'xyzzy')
        self.assertEqual(sorted(headers.get_all()), [('Foo', 'bar'), ('Quux', 'xyzzy')])

    def test_string(self):
        headers = HTTPHeaders()
        headers.add('Foo', '1')
        headers.add('Foo', '2')
        headers.add('Foo', '3')
        headers2 = HTTPHeaders.parse(str(headers))
        self.assertEqual(headers, headers2)