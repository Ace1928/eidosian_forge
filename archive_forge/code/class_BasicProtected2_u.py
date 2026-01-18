from hashlib import md5
import cherrypy
from cherrypy._cpcompat import ntob
from cherrypy.lib import auth_basic
from cherrypy.test import helper
class BasicProtected2_u:

    @cherrypy.expose
    def index(self):
        return "Hello %s, you've been authorized." % cherrypy.request.login