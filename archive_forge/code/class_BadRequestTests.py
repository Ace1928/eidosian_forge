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
class BadRequestTests(helper.CPWebCase):
    setup_server = staticmethod(setup_server)

    def test_No_CRLF(self):
        self.persistent = True
        conn = self.HTTP_CONN
        conn.send(b'GET /hello HTTP/1.1\n\n')
        response = conn.response_class(conn.sock, method='GET')
        response.begin()
        self.body = response.read()
        self.assertBody('HTTP requires CRLF terminators')
        conn.close()
        conn.connect()
        conn.send(b'GET /hello HTTP/1.1\r\n\n')
        response = conn.response_class(conn.sock, method='GET')
        response.begin()
        self.body = response.read()
        self.assertBody('HTTP requires CRLF terminators')
        conn.close()