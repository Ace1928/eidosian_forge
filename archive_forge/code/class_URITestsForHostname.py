from urllib.parse import urlparse
from twisted.trial import unittest
from twisted.web import client
class URITestsForHostname(URITests, unittest.TestCase):
    """
    Tests for L{twisted.web.client.URI} with host names.
    """
    uriHost = host = b'example.com'