import logging
import sys
import io
import cheroot.server
import cherrypy
from cherrypy._cperror import format_exc, bare_error
from cherrypy.lib import httputil
from ._cpcompat import tonative
Initialize CPHTTPServer.