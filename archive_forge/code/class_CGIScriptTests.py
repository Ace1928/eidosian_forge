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
class CGIScriptTests(_StartServerAndTearDownMixin, unittest.TestCase):
    """
    Tests for L{twcgi.CGIScript}.
    """

    def test_urlParameters(self):
        """
        If the CGI script is passed URL parameters, do not fall over,
        as per ticket 9887.
        """
        cgiFilename = self.writeCGI(URL_PARAMETER_CGI)
        portnum = self.startServer(cgiFilename)
        url = b'http://localhost:%d/cgi?param=1234' % (portnum,)
        agent = client.Agent(reactor)
        d = agent.request(b'GET', url)
        d.addCallback(client.readBody)
        d.addCallback(self._test_urlParameters_1)
        return d

    def _test_urlParameters_1(self, res):
        expected = f'param=1234{os.linesep}'
        expected = expected.encode('ascii')
        self.assertEqual(res, expected)

    def test_pathInfo(self):
        """
        L{twcgi.CGIScript.render} sets the process environment
        I{PATH_INFO} from the request path.
        """

        class FakeReactor:
            """
            A fake reactor recording the environment passed to spawnProcess.
            """

            def spawnProcess(self, process, filename, args, env, wdir):
                """
                Store the C{env} L{dict} to an instance attribute.

                @param process: Ignored
                @param filename: Ignored
                @param args: Ignored
                @param env: The environment L{dict} which will be stored
                @param wdir: Ignored
                """
                self.process_env = env
        _reactor = FakeReactor()
        resource = twcgi.CGIScript(self.mktemp(), reactor=_reactor)
        request = DummyRequest(['a', 'b'])
        request.client = address.IPv4Address('TCP', '127.0.0.1', 12345)
        _render(resource, request)
        self.assertEqual(_reactor.process_env['PATH_INFO'], '/a/b')