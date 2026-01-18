import errno
import socket
import sys
import time
import urllib.parse
from http.client import BadStatusLine, HTTPConnection, NotConnected
from cheroot.test import webtest
import cherrypy
from cherrypy._cpcompat import HTTPSConnection, ntob, tonative
from cherrypy.test import helper
class PipelineTests(helper.CPWebCase):
    setup_server = staticmethod(setup_server)

    def test_HTTP11_Timeout(self):
        if cherrypy.server.protocol_version != 'HTTP/1.1':
            return self.skip()
        self.PROTOCOL = 'HTTP/1.1'
        self.persistent = True
        conn = self.HTTP_CONN
        conn.auto_open = False
        conn.connect()
        time.sleep(timeout * 2)
        response = conn.response_class(conn.sock, method='GET')
        response.begin()
        self.assertEqual(response.status, 408)
        conn.close()
        self.persistent = True
        conn = self.HTTP_CONN
        conn.auto_open = False
        conn.connect()
        conn.send(b'GET /hello HTTP/1.1')
        conn.send(('Host: %s' % self.HOST).encode('ascii'))
        time.sleep(timeout * 2)
        response = conn.response_class(conn.sock, method='GET')
        response.begin()
        self.assertEqual(response.status, 408)
        conn.close()

    def test_HTTP11_Timeout_after_request(self):
        if cherrypy.server.protocol_version != 'HTTP/1.1':
            return self.skip()
        self.PROTOCOL = 'HTTP/1.1'
        self.persistent = True
        conn = self.HTTP_CONN
        conn.putrequest('GET', '/timeout?t=%s' % timeout, skip_host=True)
        conn.putheader('Host', self.HOST)
        conn.endheaders()
        response = conn.response_class(conn.sock, method='GET')
        response.begin()
        self.assertEqual(response.status, 200)
        self.body = response.read()
        self.assertBody(str(timeout))
        conn._output(b'GET /hello HTTP/1.1')
        conn._output(ntob('Host: %s' % self.HOST, 'ascii'))
        conn._send_output()
        response = conn.response_class(conn.sock, method='GET')
        response.begin()
        self.assertEqual(response.status, 200)
        self.body = response.read()
        self.assertBody('Hello, world!')
        time.sleep(timeout * 2)
        conn._output(b'GET /hello HTTP/1.1')
        conn._output(ntob('Host: %s' % self.HOST, 'ascii'))
        conn._send_output()
        response = conn.response_class(conn.sock, method='GET')
        msg = "Writing to timed out socket didn't fail as it should have: %s"
        try:
            response.begin()
        except Exception:
            if not isinstance(sys.exc_info()[1], (socket.error, BadStatusLine)):
                self.fail(msg % sys.exc_info()[1])
        else:
            if response.status != 408:
                self.fail(msg % response.read())
        conn.close()
        self.persistent = True
        conn = self.HTTP_CONN
        conn.putrequest('GET', '/', skip_host=True)
        conn.putheader('Host', self.HOST)
        conn.endheaders()
        response = conn.response_class(conn.sock, method='GET')
        response.begin()
        self.assertEqual(response.status, 200)
        self.body = response.read()
        self.assertBody(pov)
        conn.send(b'GET /hello HTTP/1.1')
        time.sleep(timeout * 2)
        response = conn.response_class(conn.sock, method='GET')
        try:
            response.begin()
        except Exception:
            if not isinstance(sys.exc_info()[1], (socket.error, BadStatusLine)):
                self.fail(msg % sys.exc_info()[1])
        else:
            if response.status != 408:
                self.fail(msg % response.read())
        conn.close()
        self.persistent = True
        conn = self.HTTP_CONN
        conn.putrequest('GET', '/', skip_host=True)
        conn.putheader('Host', self.HOST)
        conn.endheaders()
        response = conn.response_class(conn.sock, method='GET')
        response.begin()
        self.assertEqual(response.status, 200)
        self.body = response.read()
        self.assertBody(pov)
        conn.close()

    def test_HTTP11_pipelining(self):
        if cherrypy.server.protocol_version != 'HTTP/1.1':
            return self.skip()
        self.PROTOCOL = 'HTTP/1.1'
        self.persistent = True
        conn = self.HTTP_CONN
        conn.putrequest('GET', '/hello', skip_host=True)
        conn.putheader('Host', self.HOST)
        conn.endheaders()
        for trial in range(5):
            conn._output(b'GET /hello HTTP/1.1')
            conn._output(ntob('Host: %s' % self.HOST, 'ascii'))
            conn._send_output()
            response = conn.response_class(conn.sock, method='GET')
            response.fp = conn.sock.makefile('rb', 0)
            response.begin()
            body = response.read(13)
            self.assertEqual(response.status, 200)
            self.assertEqual(body, b'Hello, world!')
        response = conn.response_class(conn.sock, method='GET')
        response.begin()
        body = response.read()
        self.assertEqual(response.status, 200)
        self.assertEqual(body, b'Hello, world!')
        conn.close()

    def test_100_Continue(self):
        if cherrypy.server.protocol_version != 'HTTP/1.1':
            return self.skip()
        self.PROTOCOL = 'HTTP/1.1'
        self.persistent = True
        conn = self.HTTP_CONN
        try:
            conn.putrequest('POST', '/upload', skip_host=True)
            conn.putheader('Host', self.HOST)
            conn.putheader('Content-Type', 'text/plain')
            conn.putheader('Content-Length', '4')
            conn.endheaders()
            conn.send(ntob("d'oh"))
            response = conn.response_class(conn.sock, method='POST')
            version, status, reason = response._read_status()
            self.assertNotEqual(status, 100)
        finally:
            conn.close()
        try:
            conn.connect()
            conn.putrequest('POST', '/upload', skip_host=True)
            conn.putheader('Host', self.HOST)
            conn.putheader('Content-Type', 'text/plain')
            conn.putheader('Content-Length', '17')
            conn.putheader('Expect', '100-continue')
            conn.endheaders()
            response = conn.response_class(conn.sock, method='POST')
            version, status, reason = response._read_status()
            self.assertEqual(status, 100)
            while True:
                line = response.fp.readline().strip()
                if line:
                    self.fail('100 Continue should not output any headers. Got %r' % line)
                else:
                    break
            body = b'I am a small file'
            conn.send(body)
            response.begin()
            self.status, self.headers, self.body = webtest.shb(response)
            self.assertStatus(200)
            self.assertBody("thanks for '%s'" % body)
        finally:
            conn.close()