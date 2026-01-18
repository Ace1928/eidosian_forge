from twisted.python import urlpath
from twisted.trial import unittest
class StringURLPathTests(_BaseURLPathTests, unittest.TestCase):
    """
    Tests for interacting with a L{URLPath} created with C{fromString} and a
    L{str} argument.
    """

    def setUp(self):
        self.path = urlpath.URLPath.fromString('http://example.com/foo/bar?yes=no&no=yes#footer')

    def test_mustBeStr(self):
        """
        C{URLPath.fromString} must take a L{str} or L{str} argument.
        """
        with self.assertRaises(ValueError):
            urlpath.URLPath.fromString(None)
        with self.assertRaises(ValueError):
            urlpath.URLPath.fromString(b'someurl')