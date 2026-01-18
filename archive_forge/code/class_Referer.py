import os
import cherrypy
from cherrypy import tools
from cherrypy.test import helper
class Referer:

    @cherrypy.expose
    def accept(self):
        return 'Accepted!'
    reject = accept