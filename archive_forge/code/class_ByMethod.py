import sys
import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy._cptree import Application
from cherrypy.test import helper
@cherrypy.expose
class ByMethod:

    def __init__(self, *things):
        self.things = list(things)

    def GET(self):
        return repr(self.things)

    def POST(self, thing):
        self.things.append(thing)