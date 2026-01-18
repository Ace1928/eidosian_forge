import sys as _sys
import io
import cherrypy as _cherrypy
from cherrypy._cpcompat import ntou
from cherrypy import _cperror
from cherrypy.lib import httputil
from cherrypy.lib import is_closable_iterator
class ExceptionTrapper(object):
    """WSGI middleware that traps exceptions."""

    def __init__(self, nextapp, throws=(KeyboardInterrupt, SystemExit)):
        self.nextapp = nextapp
        self.throws = throws

    def __call__(self, environ, start_response):
        return _TrappedResponse(self.nextapp, environ, start_response, self.throws)