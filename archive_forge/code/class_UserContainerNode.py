import cherrypy
from cherrypy.test import helper
@cherrypy.expose
class UserContainerNode(object):

    def POST(self, name):
        """
            Allow the creation of a new Object
            """
        return 'POST %d' % make_user(name)

    def GET(self):
        return str(sorted(user_lookup.keys()))

    def dynamic_dispatch(self, vpath):
        try:
            id = int(vpath[0])
        except (ValueError, IndexError):
            return None
        return UserInstanceNode(id)