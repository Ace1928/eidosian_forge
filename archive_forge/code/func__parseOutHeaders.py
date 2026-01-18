from twisted.internet.testing import MemoryReactor, StringTransportWithDisconnection
from twisted.trial.unittest import TestCase
from twisted.web.proxy import (
from twisted.web.resource import Resource
from twisted.web.server import Site
from twisted.web.test.test_web import DummyRequest
def _parseOutHeaders(self, content):
    """
        Parse the headers out of some web content.

        @param content: Bytes received from a web server.
        @return: A tuple of (requestLine, headers, body). C{headers} is a dict
            of headers, C{requestLine} is the first line (e.g. "POST /foo ...")
            and C{body} is whatever is left.
        """
    headers, body = content.split(b'\r\n\r\n')
    headers = headers.split(b'\r\n')
    requestLine = headers.pop(0)
    return (requestLine, dict((header.split(b': ') for header in headers)), body)