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
class TestUrlConcat(unittest.TestCase):

    def test_url_concat_no_query_params(self):
        url = url_concat('https://localhost/path', [('y', 'y'), ('z', 'z')])
        self.assertEqual(url, 'https://localhost/path?y=y&z=z')

    def test_url_concat_encode_args(self):
        url = url_concat('https://localhost/path', [('y', '/y'), ('z', 'z')])
        self.assertEqual(url, 'https://localhost/path?y=%2Fy&z=z')

    def test_url_concat_trailing_q(self):
        url = url_concat('https://localhost/path?', [('y', 'y'), ('z', 'z')])
        self.assertEqual(url, 'https://localhost/path?y=y&z=z')

    def test_url_concat_q_with_no_trailing_amp(self):
        url = url_concat('https://localhost/path?x', [('y', 'y'), ('z', 'z')])
        self.assertEqual(url, 'https://localhost/path?x=&y=y&z=z')

    def test_url_concat_trailing_amp(self):
        url = url_concat('https://localhost/path?x&', [('y', 'y'), ('z', 'z')])
        self.assertEqual(url, 'https://localhost/path?x=&y=y&z=z')

    def test_url_concat_mult_params(self):
        url = url_concat('https://localhost/path?a=1&b=2', [('y', 'y'), ('z', 'z')])
        self.assertEqual(url, 'https://localhost/path?a=1&b=2&y=y&z=z')

    def test_url_concat_no_params(self):
        url = url_concat('https://localhost/path?r=1&t=2', [])
        self.assertEqual(url, 'https://localhost/path?r=1&t=2')

    def test_url_concat_none_params(self):
        url = url_concat('https://localhost/path?r=1&t=2', None)
        self.assertEqual(url, 'https://localhost/path?r=1&t=2')

    def test_url_concat_with_frag(self):
        url = url_concat('https://localhost/path#tab', [('y', 'y')])
        self.assertEqual(url, 'https://localhost/path?y=y#tab')

    def test_url_concat_multi_same_params(self):
        url = url_concat('https://localhost/path', [('y', 'y1'), ('y', 'y2')])
        self.assertEqual(url, 'https://localhost/path?y=y1&y=y2')

    def test_url_concat_multi_same_query_params(self):
        url = url_concat('https://localhost/path?r=1&r=2', [('y', 'y')])
        self.assertEqual(url, 'https://localhost/path?r=1&r=2&y=y')

    def test_url_concat_dict_params(self):
        url = url_concat('https://localhost/path', dict(y='y'))
        self.assertEqual(url, 'https://localhost/path?y=y')