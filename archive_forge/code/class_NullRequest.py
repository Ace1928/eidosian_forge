import getopt
import os
import re
import sys
import time
import cherrypy
from cherrypy import _cperror, _cpmodpy
from cherrypy.lib import httputil
class NullRequest:
    """A null HTTP request class, returning 200 and an empty body."""

    def __init__(self, local, remote, scheme='http'):
        pass

    def close(self):
        pass

    def run(self, method, path, query_string, protocol, headers, rfile):
        cherrypy.response.status = '200 OK'
        cherrypy.response.header_list = [('Content-Type', 'text/html'), ('Server', 'Null CherryPy'), ('Date', httputil.HTTPDate()), ('Content-Length', '0')]
        cherrypy.response.body = ['']
        return cherrypy.response