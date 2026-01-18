import sys
import time
import collections
import operator
from http.cookies import SimpleCookie, CookieError
import uuid
from more_itertools import consume
import cherrypy
from cherrypy._cpcompat import ntob
from cherrypy import _cpreqbody
from cherrypy._cperror import format_exc, bare_error
from cherrypy.lib import httputil, reprconf, encoding
def _flush_body(self):
    """
        Discard self.body but consume any generator such that
        any finalization can occur, such as is required by
        caching.tee_output().
        """
    consume(iter(self.body))