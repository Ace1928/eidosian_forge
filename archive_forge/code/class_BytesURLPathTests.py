from twisted.python import urlpath
from twisted.trial import unittest
class BytesURLPathTests(_BaseURLPathTests, unittest.TestCase):
    """
    Tests for interacting with a L{URLPath} created with C{fromBytes}.
    """

    def setUp(self):
        self.path = urlpath.URLPath.fromBytes(b'http://example.com/foo/bar?yes=no&no=yes#footer')

    def test_mustBeBytes(self):
        """
        L{URLPath.fromBytes} must take a L{bytes} argument.
        """
        with self.assertRaises(ValueError):
            urlpath.URLPath.fromBytes(None)
        with self.assertRaises(ValueError):
            urlpath.URLPath.fromBytes('someurl')

    def test_withoutArguments(self):
        """
        An instantiation with no arguments creates a usable L{URLPath} with
        default arguments.
        """
        url = urlpath.URLPath()
        self.assertEqual(str(url), 'http://localhost/')

    def test_partialArguments(self):
        """
        Leaving some optional arguments unfilled makes a L{URLPath} with those
        optional arguments filled with defaults.
        """
        url = urlpath.URLPath.fromBytes(b'http://google.com')
        self.assertEqual(url.scheme, b'http')
        self.assertEqual(url.netloc, b'google.com')
        self.assertEqual(url.path, b'/')
        self.assertEqual(url.fragment, b'')
        self.assertEqual(url.query, b'')
        self.assertEqual(str(url), 'http://google.com/')

    def test_nonASCIIBytes(self):
        """
        L{URLPath.fromBytes} can interpret non-ASCII bytes as percent-encoded
        """
        url = urlpath.URLPath.fromBytes(b'http://example.com/\xff\x00')
        self.assertEqual(str(url), 'http://example.com/%FF%00')