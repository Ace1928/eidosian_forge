import cherrypy
from cherrypy.test import helper
class DecoratedPopArgs:
    """Test _cp_dispatch with @cherrypy.popargs."""

    @cherrypy.expose
    def index(self):
        return 'no params'

    @cherrypy.expose
    def hi(self):
        return "hi was not interpreted as 'a' param"