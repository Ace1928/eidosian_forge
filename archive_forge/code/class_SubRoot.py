import cherrypy
from cherrypy.test import helper
class SubRoot:

    @cherrypy.expose
    def index(self):
        return 'SubRoot index'

    @cherrypy.expose
    def default(self, *args):
        return 'SubRoot %s' % (args,)

    @cherrypy.expose
    def handler(self):
        return 'SubRoot handler'

    def _cp_dispatch(self, vpath):
        return subsubnodes.get(vpath[0], None)