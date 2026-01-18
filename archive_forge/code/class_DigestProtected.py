import cherrypy
from cherrypy.lib import auth_digest
from cherrypy._cpcompat import ntob
from cherrypy.test import helper
class DigestProtected:

    @cherrypy.expose
    def index(self, *args, **kwargs):
        return "Hello %s, you've been authorized." % cherrypy.request.login