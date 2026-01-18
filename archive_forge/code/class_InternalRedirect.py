import os
import sys
import types
import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy import _cptools, tools
from cherrypy.lib import httputil, static
from cherrypy.test._test_decorators import ExposeExamples
from cherrypy.test import helper
class InternalRedirect(Test):

    def index(self):
        raise cherrypy.InternalRedirect('/')

    @cherrypy.expose
    @cherrypy.config(**{'hooks.before_error_response': redir_custom})
    def choke(self):
        return 3 / 0

    def relative(self, a, b):
        raise cherrypy.InternalRedirect('cousin?t=6')

    def cousin(self, t):
        assert cherrypy.request.prev.closed
        return cherrypy.request.prev.query_string

    def petshop(self, user_id):
        if user_id == 'parrot':
            raise cherrypy.InternalRedirect('/image/getImagesByUser?user_id=slug')
        elif user_id == 'terrier':
            raise cherrypy.InternalRedirect('/image/getImagesByUser?user_id=fish')
        else:
            raise cherrypy.InternalRedirect('/image/getImagesByUser?user_id=%s' % str(user_id))

    def secure(self):
        return 'Welcome!'
    secure = tools.login_redir()(secure)

    def login(self):
        return 'Please log in'

    def custom_err(self):
        return 'Something went horribly wrong.'

    @cherrypy.config(**{'hooks.before_request_body': redir_custom})
    def early_ir(self, arg):
        return 'whatever'