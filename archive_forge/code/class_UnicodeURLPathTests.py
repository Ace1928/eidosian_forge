from twisted.python import urlpath
from twisted.trial import unittest
class UnicodeURLPathTests(_BaseURLPathTests, unittest.TestCase):
    """
    Tests for interacting with a L{URLPath} created with C{fromString} and a
    L{str} argument.
    """

    def setUp(self):
        self.path = urlpath.URLPath.fromString('http://example.com/foo/bar?yes=no&no=yes#footer')

    def test_nonASCIICharacters(self):
        """
        L{URLPath.fromString} can load non-ASCII characters.
        """
        url = urlpath.URLPath.fromString('http://example.com/ÿ\x00')
        self.assertEqual(str(url), 'http://example.com/%C3%BF%00')