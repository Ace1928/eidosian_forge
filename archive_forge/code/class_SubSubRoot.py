import cherrypy
from cherrypy.test import helper
class SubSubRoot:

    @cherrypy.expose
    def index(self):
        return 'SubSubRoot index'

    @cherrypy.expose
    def default(self, *args):
        return 'SubSubRoot default'

    @cherrypy.expose
    def handler(self):
        return 'SubSubRoot handler'

    @cherrypy.expose
    def dispatch(self):
        return 'SubSubRoot dispatch'