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
class LazyUUID4(object):

    def __str__(self):
        """Return UUID4 and keep it for future calls."""
        return str(self.uuid4)

    @property
    def uuid4(self):
        """Provide unique id on per-request basis using UUID4.

        It's evaluated lazily on render.
        """
        try:
            self._uuid4
        except AttributeError:
            self._uuid4 = uuid.uuid4()
        return self._uuid4