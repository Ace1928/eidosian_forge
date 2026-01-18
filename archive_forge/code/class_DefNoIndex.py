import sys
import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy._cptree import Application
from cherrypy.test import helper
class DefNoIndex:

    @cherrypy.expose
    def default(self, *args):
        raise cherrypy.HTTPRedirect('contact')