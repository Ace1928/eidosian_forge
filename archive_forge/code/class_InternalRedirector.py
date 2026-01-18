import sys as _sys
import io
import cherrypy as _cherrypy
from cherrypy._cpcompat import ntou
from cherrypy import _cperror
from cherrypy.lib import httputil
from cherrypy.lib import is_closable_iterator
class InternalRedirector(object):
    """WSGI middleware that handles raised cherrypy.InternalRedirect."""

    def __init__(self, nextapp, recursive=False):
        self.nextapp = nextapp
        self.recursive = recursive

    def __call__(self, environ, start_response):
        redirections = []
        while True:
            environ = environ.copy()
            try:
                return self.nextapp(environ, start_response)
            except _cherrypy.InternalRedirect:
                ir = _sys.exc_info()[1]
                sn = environ.get('SCRIPT_NAME', '')
                path = environ.get('PATH_INFO', '')
                qs = environ.get('QUERY_STRING', '')
                old_uri = sn + path
                if qs:
                    old_uri += '?' + qs
                redirections.append(old_uri)
                if not self.recursive:
                    new_uri = sn + ir.path
                    if ir.query_string:
                        new_uri += '?' + ir.query_string
                    if new_uri in redirections:
                        ir.request.close()
                        tmpl = 'InternalRedirector visited the same URL twice: %r'
                        raise RuntimeError(tmpl % new_uri)
                environ['REQUEST_METHOD'] = 'GET'
                environ['PATH_INFO'] = ir.path
                environ['QUERY_STRING'] = ir.query_string
                environ['wsgi.input'] = io.BytesIO()
                environ['CONTENT_LENGTH'] = '0'
                environ['cherrypy.previous_request'] = ir.request