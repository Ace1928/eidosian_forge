import io
import os
import sys
import unittest
import cherrypy
from cherrypy.test import helper
class ConfigTests(helper.CPWebCase):
    setup_server = staticmethod(setup_server)

    def testConfig(self):
        tests = [('/', 'nex', 'None'), ('/', 'foo', 'this'), ('/', 'bar', 'that'), ('/xyz', 'foo', 'this'), ('/foo/', 'foo', 'this2'), ('/foo/', 'bar', 'that'), ('/foo/', 'bax', 'None'), ('/foo/bar', 'baz', "'that2'"), ('/foo/nex', 'baz', 'that2'), ('/another/', 'foo', 'None')]
        for path, key, expected in tests:
            self.getPage(path + '?key=' + key)
            self.assertBody(expected)
        expectedconf = {'tools.log_headers.on': False, 'tools.log_tracebacks.on': True, 'request.show_tracebacks': True, 'log.screen': False, 'environment': 'test_suite', 'engine.autoreload.on': False, 'luxuryyacht': 'throatwobblermangrove', 'bar': 'that', 'baz': 'that2', 'foo': 'this3', 'bax': 'this4'}
        for key, expected in expectedconf.items():
            self.getPage('/foo/bar?key=' + key)
            self.assertBody(repr(expected))

    def testUnrepr(self):
        self.getPage('/repr?key=neg')
        self.assertBody('-1234')
        self.getPage('/repr?key=filename')
        self.assertBody(repr(os.path.join(sys.prefix, 'hello.py')))
        self.getPage('/repr?key=thing1')
        self.assertBody(repr(cherrypy.lib.httputil.response_codes[404]))
        if not getattr(cherrypy.server, 'using_apache', False):
            self.getPage('/repr?key=thing2')
            from cherrypy.tutorial import thing2
            self.assertBody(repr(thing2))
        self.getPage('/repr?key=complex')
        self.assertBody('(3+2j)')
        self.getPage('/repr?key=mul')
        self.assertBody('18')
        self.getPage('/repr?key=stradd')
        self.assertBody(repr('112233'))

    def testRespNamespaces(self):
        self.getPage('/foo/silly')
        self.assertHeader('X-silly', 'sillyval')
        self.assertBody('Hello world')

    def testCustomNamespaces(self):
        self.getPage('/raw/incr?num=12')
        self.assertBody('13')
        self.getPage('/dbscheme')
        self.assertBody('sqlite///memory')

    def testHandlerToolConfigOverride(self):
        self.getPage('/favicon.ico')
        with open(os.path.join(localDir, 'static/dirback.jpg'), 'rb') as tf:
            self.assertBody(tf.read())

    def test_request_body_namespace(self):
        self.getPage('/plain', method='POST', headers=[('Content-Type', 'application/x-www-form-urlencoded'), ('Content-Length', '13')], body=b'\xff\xfex\x00=\xff\xfea\x00b\x00c\x00')
        self.assertBody('abc')