import cherrypy
from cherrypy.test import helper
@cherrypy.expose
class UserInstanceNode(object):

    def __init__(self, id):
        self.id = id
        self.user = user_lookup.get(id, None)
        if not self.user and cherrypy.request.method != 'PUT':
            raise cherrypy.HTTPError(404)

    def GET(self, *args, **kwargs):
        """
            Return the appropriate representation of the instance.
            """
        return str(self.user)

    def POST(self, name):
        """
            Update the fields of the user instance.
            """
        self.user.name = name
        return 'POST %d' % self.user.id

    def PUT(self, name):
        """
            Create a new user with the specified id, or edit it if it already
            exists
            """
        if self.user:
            self.user.name = name
            return 'PUT %d' % self.user.id
        else:
            return 'PUT %d' % make_user(name, self.id)

    def DELETE(self):
        """
            Delete the user specified at the id.
            """
        id = self.user.id
        del user_lookup[self.user.id]
        del self.user
        return 'DELETE %d' % id