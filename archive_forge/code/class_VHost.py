import os
import cherrypy
from cherrypy.test import helper
class VHost:

    def __init__(self, sitename):
        self.sitename = sitename

    @cherrypy.expose
    def index(self):
        return 'Welcome to %s' % self.sitename

    @cherrypy.expose
    def vmethod(self, value):
        return 'You sent %s' % value

    @cherrypy.expose
    def url(self):
        return cherrypy.url('nextpage')
    static = cherrypy.tools.staticdir.handler(section='/static', dir=curdir)