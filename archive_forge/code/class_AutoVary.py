import os
import cherrypy
from cherrypy import tools
from cherrypy.test import helper
class AutoVary:

    @cherrypy.expose
    def index(self):
        cherrypy.request.headers.get('Accept-Encoding')
        cherrypy.request.headers['Host']
        'If-Modified-Since' in cherrypy.request.headers
        'Range' in cherrypy.request.headers
        tools.accept.callable(['text/html', 'text/plain'])
        return 'Hello, world!'