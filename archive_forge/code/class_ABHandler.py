import cherrypy
from cherrypy.test import helper
class ABHandler:

    class CustomDispatch:

        @cherrypy.expose
        def index(self, a, b):
            return 'custom'

    def _cp_dispatch(self, vpath):
        """Make sure that if we don't pop anything from vpath,
            processing still works.
            """
        return self.CustomDispatch()

    @cherrypy.expose
    def index(self, a, b=None):
        body = ['a:' + str(a)]
        if b is not None:
            body.append(',b:' + str(b))
        return ''.join(body)

    @cherrypy.expose
    def delete(self, a, b):
        return 'deleting ' + str(a) + ' and ' + str(b)