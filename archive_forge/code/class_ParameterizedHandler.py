import cherrypy
from cherrypy.test import helper
class ParameterizedHandler:
    """Special handler created for each request"""

    def __init__(self, a):
        self.a = a

    @cherrypy.expose
    def index(self):
        if 'a' in cherrypy.request.params:
            raise Exception('Parameterized handler argument ended up in request.params')
        return self.a