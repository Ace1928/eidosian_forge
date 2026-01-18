import cherrypy
from cherrypy.lib import auth_digest
from cherrypy._cpcompat import ntob
from cherrypy.test import helper
def _fetch_users():
    return {'test': 'test', '☃йюзер': 'їпароль'}