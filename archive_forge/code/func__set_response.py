import sys
from xmlrpc.client import (
import cherrypy
from cherrypy._cpcompat import ntob
def _set_response(body):
    """Set up HTTP status, headers and body within CherryPy."""
    response = cherrypy.response
    response.status = '200 OK'
    response.body = ntob(body, 'utf-8')
    response.headers['Content-Type'] = 'text/xml'
    response.headers['Content-Length'] = len(body)