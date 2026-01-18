import os
import traceback
import unittest
from tornado.escape import utf8, native_str, to_unicode
from tornado.template import Template, DictLoader, ParseError, Loader
from tornado.util import ObjectDict
import typing  # noqa: F401
class ParseErrorDetailTest(unittest.TestCase):

    def test_details(self):
        loader = DictLoader({'foo.html': '\n\n{{'})
        with self.assertRaises(ParseError) as cm:
            loader.load('foo.html')
        self.assertEqual('Missing end expression }} at foo.html:3', str(cm.exception))
        self.assertEqual('foo.html', cm.exception.filename)
        self.assertEqual(3, cm.exception.lineno)

    def test_custom_parse_error(self):
        self.assertEqual('asdf at None:0', str(ParseError('asdf')))