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
class ParseCookieTest(unittest.TestCase):

    def test_python_cookies(self):
        """
        Test cases copied from Python's Lib/test/test_http_cookies.py
        """
        self.assertEqual(parse_cookie('chips=ahoy; vienna=finger'), {'chips': 'ahoy', 'vienna': 'finger'})
        self.assertEqual(parse_cookie('keebler="E=mc2; L=\\"Loves\\"; fudge=\\012;"'), {'keebler': '"E=mc2', 'L': '\\"Loves\\"', 'fudge': '\\012', '': '"'})
        self.assertEqual(parse_cookie('keebler=E=mc2'), {'keebler': 'E=mc2'})
        self.assertEqual(parse_cookie('key:term=value:term'), {'key:term': 'value:term'})
        self.assertEqual(parse_cookie('a=b; c=[; d=r; f=h'), {'a': 'b', 'c': '[', 'd': 'r', 'f': 'h'})

    def test_cookie_edgecases(self):
        self.assertEqual(parse_cookie('a=b; Domain=example.com'), {'a': 'b', 'Domain': 'example.com'})
        self.assertEqual(parse_cookie('a=b; h=i; a=c'), {'a': 'c', 'h': 'i'})

    def test_invalid_cookies(self):
        """
        Cookie strings that go against RFC6265 but browsers will send if set
        via document.cookie.
        """
        self.assertIn('django_language', parse_cookie('abc=def; unnamed; django_language=en').keys())
        self.assertEqual(parse_cookie('a=b; "; c=d'), {'a': 'b', '': '"', 'c': 'd'})
        self.assertEqual(parse_cookie('a b c=d e = f; gh=i'), {'a b c': 'd e = f', 'gh': 'i'})
        self.assertEqual(parse_cookie('a   b,c<>@:/[]?{}=d  "  =e,f g'), {'a   b,c<>@:/[]?{}': 'd  "  =e,f g'})
        self.assertEqual(parse_cookie('saint=André Bessette'), {'saint': native_str('André Bessette')})
        self.assertEqual(parse_cookie('  =  b  ;  ;  =  ;   c  =  ;  '), {'': 'b', 'c': ''})