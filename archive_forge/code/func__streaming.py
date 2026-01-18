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
def _streaming(self, set_cl):
    if cherrypy.server.protocol_version == 'HTTP/1.1':
        self.PROTOCOL = 'HTTP/1.1'
        self.persistent = True
        self.getPage('/')
        self.assertStatus('200 OK')
        self.assertBody(pov)
        self.assertNoHeader('Connection')
        if set_cl:
            self.getPage('/stream?set_cl=Yes')
            self.assertHeader('Content-Length')
            self.assertNoHeader('Connection', 'close')
            self.assertNoHeader('Transfer-Encoding')
            self.assertStatus('200 OK')
            self.assertBody('0123456789')
        else:
            self.getPage('/stream')
            self.assertNoHeader('Content-Length')
            self.assertStatus('200 OK')
            self.assertBody('0123456789')
            chunked_response = False
            for k, v in self.headers:
                if k.lower() == 'transfer-encoding':
                    if str(v) == 'chunked':
                        chunked_response = True
            if chunked_response:
                self.assertNoHeader('Connection', 'close')
            else:
                self.assertHeader('Connection', 'close')
                self.assertRaises(NotConnected, self.getPage, '/')
            self.getPage('/stream', method='HEAD')
            self.assertStatus('200 OK')
            self.assertBody('')
            self.assertNoHeader('Transfer-Encoding')
    else:
        self.PROTOCOL = 'HTTP/1.0'
        self.persistent = True
        self.getPage('/', headers=[('Connection', 'Keep-Alive')])
        self.assertStatus('200 OK')
        self.assertBody(pov)
        self.assertHeader('Connection', 'Keep-Alive')
        if set_cl:
            self.getPage('/stream?set_cl=Yes', headers=[('Connection', 'Keep-Alive')])
            self.assertHeader('Content-Length')
            self.assertHeader('Connection', 'Keep-Alive')
            self.assertNoHeader('Transfer-Encoding')
            self.assertStatus('200 OK')
            self.assertBody('0123456789')
        else:
            self.getPage('/stream', headers=[('Connection', 'Keep-Alive')])
            self.assertStatus('200 OK')
            self.assertBody('0123456789')
            self.assertNoHeader('Content-Length')
            self.assertNoHeader('Connection', 'Keep-Alive')
            self.assertNoHeader('Transfer-Encoding')
            self.assertRaises(NotConnected, self.getPage, '/')