from functools import wraps
import os
import sys
import types
import uuid
from http.client import IncompleteRead
import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy.lib import httputil
from cherrypy.test import helper
class Divorce(Test):
    """HTTP Method handlers shouldn't collide with normal method names.
            For example, a GET-handler shouldn't collide with a method named
            'get'.

            If you build HTTP method dispatching into CherryPy, rewrite this
            class to use your new dispatch mechanism and make sure that:
                "GET /divorce HTTP/1.1" maps to divorce.index() and
                "GET /divorce/get?ID=13 HTTP/1.1" maps to divorce.get()
            """
    documents = {}

    @cherrypy.expose
    def index(self):
        yield '<h1>Choose your document</h1>\n'
        yield '<ul>\n'
        for id, contents in self.documents.items():
            yield ("    <li><a href='/divorce/get?ID=%s'>%s</a>: %s</li>\n" % (id, id, contents))
        yield '</ul>'

    @cherrypy.expose
    def get(self, ID):
        return 'Divorce document %s: %s' % (ID, self.documents.get(ID, 'empty'))