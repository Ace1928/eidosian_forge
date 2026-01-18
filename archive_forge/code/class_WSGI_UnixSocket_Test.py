import os
import socket
import atexit
import tempfile
from http.client import HTTPConnection
import pytest
import cherrypy
from cherrypy.test import helper
@pytest.mark.skipif("sys.platform == 'win32'")
class WSGI_UnixSocket_Test(helper.CPWebCase):
    """
    Test basic behavior on a cherrypy wsgi server listening
    on a unix socket.

    It exercises the config option `server.socket_file`.
    """
    HTTP_CONN = USocketHTTPConnection(USOCKET_PATH)

    @staticmethod
    def setup_server():

        class Root(object):

            @cherrypy.expose
            def index(self):
                return 'Test OK'

            @cherrypy.expose
            def error(self):
                raise Exception('Invalid page')
        config = {'server.socket_file': USOCKET_PATH}
        cherrypy.config.update(config)
        cherrypy.tree.mount(Root())

    def tearDown(self):
        cherrypy.config.update({'server.socket_file': None})

    def test_simple_request(self):
        self.getPage('/')
        self.assertStatus('200 OK')
        self.assertInBody('Test OK')

    def test_not_found(self):
        self.getPage('/invalid_path')
        self.assertStatus('404 Not Found')

    def test_internal_error(self):
        self.getPage('/error')
        self.assertStatus('500 Internal Server Error')
        self.assertInBody('Invalid page')