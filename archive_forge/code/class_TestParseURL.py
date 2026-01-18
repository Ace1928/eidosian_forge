import os
import sys
from .. import osutils, urlutils
from ..errors import PathNotChild
from . import TestCase, TestCaseInTempDir, TestSkipped, features
class TestParseURL(TestCase):

    def test_parse_simple(self):
        parsed = urlutils.parse_url('http://example.com:80/one')
        self.assertEqual(('http', None, None, 'example.com', 80, '/one'), parsed)

    def test_ipv6(self):
        parsed = urlutils.parse_url('http://[1:2:3::40]/one')
        self.assertEqual(('http', None, None, '1:2:3::40', None, '/one'), parsed)

    def test_ipv6_port(self):
        parsed = urlutils.parse_url('http://[1:2:3::40]:80/one')
        self.assertEqual(('http', None, None, '1:2:3::40', 80, '/one'), parsed)