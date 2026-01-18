import gzip
import io
from unittest import mock
from http.client import IncompleteRead
from urllib.parse import quote as url_quote
import cherrypy
from cherrypy._cpcompat import ntob, ntou
from cherrypy.test import helper
class GZIP:

    @cherrypy.expose
    def index(self):
        yield 'Hello, world'

    @cherrypy.expose
    @cherrypy.config(**{'tools.encode.on': False})
    def noshow(self):
        raise IndexError()
        yield 'Here be dragons'

    @cherrypy.expose
    @cherrypy.config(**{'response.stream': True})
    def noshow_stream(self):
        raise IndexError()
        yield 'Here be dragons'