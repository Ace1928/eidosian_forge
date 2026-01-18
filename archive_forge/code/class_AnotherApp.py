import sys
import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy._cptree import Application
from cherrypy.test import helper
@cherrypy.expose
class AnotherApp:

    def GET(self):
        return 'milk'