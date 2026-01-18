import json
import os
import sys
from io import BytesIO
from twisted.internet import address, error, interfaces, reactor
from twisted.internet.error import ConnectionLost
from twisted.python import failure, log, util
from twisted.trial import unittest
from twisted.web import client, http, http_headers, resource, server, twcgi
from twisted.web.http import INTERNAL_SERVER_ERROR, NOT_FOUND
from twisted.web.test._util import _render
from twisted.web.test.requesthelper import DummyChannel, DummyRequest
import os, sys
import sys
import json
import os
import os
class CGIDirectoryTests(unittest.TestCase):
    """
    Tests for L{twcgi.CGIDirectory}.
    """

    def test_render(self):
        """
        L{twcgi.CGIDirectory.render} sets the HTTP response code to I{NOT
        FOUND}.
        """
        resource = twcgi.CGIDirectory(self.mktemp())
        request = DummyRequest([''])
        d = _render(resource, request)

        def cbRendered(ignored):
            self.assertEqual(request.responseCode, NOT_FOUND)
        d.addCallback(cbRendered)
        return d

    def test_notFoundChild(self):
        """
        L{twcgi.CGIDirectory.getChild} returns a resource which renders an
        response with the HTTP I{NOT FOUND} status code if the indicated child
        does not exist as an entry in the directory used to initialized the
        L{twcgi.CGIDirectory}.
        """
        path = self.mktemp()
        os.makedirs(path)
        resource = twcgi.CGIDirectory(path)
        request = DummyRequest(['foo'])
        child = resource.getChild('foo', request)
        d = _render(child, request)

        def cbRendered(ignored):
            self.assertEqual(request.responseCode, NOT_FOUND)
        d.addCallback(cbRendered)
        return d