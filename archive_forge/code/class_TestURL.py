import os
import sys
from .. import osutils, urlutils
from ..errors import PathNotChild
from . import TestCase, TestCaseInTempDir, TestSkipped, features
class TestURL(TestCase):

    def test_parse_simple(self):
        parsed = urlutils.URL.from_string('http://example.com:80/one')
        self.assertEqual('http', parsed.scheme)
        self.assertIs(None, parsed.user)
        self.assertIs(None, parsed.password)
        self.assertEqual('example.com', parsed.host)
        self.assertEqual(80, parsed.port)
        self.assertEqual('/one', parsed.path)

    def test_ipv6(self):
        parsed = urlutils.URL.from_string('http://[1:2:3::40]/one')
        self.assertEqual('http', parsed.scheme)
        self.assertIs(None, parsed.port)
        self.assertIs(None, parsed.user)
        self.assertIs(None, parsed.password)
        self.assertEqual('1:2:3::40', parsed.host)
        self.assertEqual('/one', parsed.path)

    def test_ipv6_port(self):
        parsed = urlutils.URL.from_string('http://[1:2:3::40]:80/one')
        self.assertEqual('http', parsed.scheme)
        self.assertEqual('1:2:3::40', parsed.host)
        self.assertIs(None, parsed.user)
        self.assertIs(None, parsed.password)
        self.assertEqual(80, parsed.port)
        self.assertEqual('/one', parsed.path)

    def test_quoted(self):
        parsed = urlutils.URL.from_string('http://ro%62ey:h%40t@ex%41mple.com:2222/path')
        self.assertEqual(parsed.quoted_host, 'ex%41mple.com')
        self.assertEqual(parsed.host, 'exAmple.com')
        self.assertEqual(parsed.port, 2222)
        self.assertEqual(parsed.quoted_user, 'ro%62ey')
        self.assertEqual(parsed.user, 'robey')
        self.assertEqual(parsed.quoted_password, 'h%40t')
        self.assertEqual(parsed.password, 'h@t')
        self.assertEqual(parsed.path, '/path')

    def test_eq(self):
        parsed1 = urlutils.URL.from_string('http://[1:2:3::40]:80/one')
        parsed2 = urlutils.URL.from_string('http://[1:2:3::40]:80/one')
        self.assertEqual(parsed1, parsed2)
        self.assertEqual(parsed1, parsed1)
        parsed2.path = '/two'
        self.assertNotEqual(parsed1, parsed2)

    def test_repr(self):
        parsed = urlutils.URL.from_string('http://[1:2:3::40]:80/one')
        self.assertEqual("<URL('http', None, None, '1:2:3::40', 80, '/one')>", repr(parsed))

    def test_str(self):
        parsed = urlutils.URL.from_string('http://[1:2:3::40]:80/one')
        self.assertEqual('http://[1:2:3::40]:80/one', str(parsed))

    def test__combine_paths(self):
        combine = urlutils.URL._combine_paths
        self.assertEqual('/home/sarah/project/foo', combine('/home/sarah', 'project/foo'))
        self.assertEqual('/etc', combine('/home/sarah', '../../etc'))
        self.assertEqual('/etc', combine('/home/sarah', '../../../etc'))
        self.assertEqual('/etc', combine('/home/sarah', '/etc'))

    def test_clone(self):
        url = urlutils.URL.from_string('http://[1:2:3::40]:80/one')
        url1 = url.clone('two')
        self.assertEqual('/one/two', url1.path)
        url2 = url.clone('/two')
        self.assertEqual('/two', url2.path)
        url3 = url.clone()
        self.assertIsNot(url, url3)
        self.assertEqual(url, url3)

    def test_parse_empty_port(self):
        parsed = urlutils.URL.from_string('http://example.com:/one')
        self.assertEqual('http', parsed.scheme)
        self.assertIs(None, parsed.user)
        self.assertIs(None, parsed.password)
        self.assertEqual('example.com', parsed.host)
        self.assertIs(None, parsed.port)
        self.assertEqual('/one', parsed.path)