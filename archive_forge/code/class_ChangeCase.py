import cherrypy
from cherrypy.test import helper
class ChangeCase(object):

    def __init__(self, app, to=None):
        self.app = app
        self.to = to

    def __call__(self, environ, start_response):
        res = self.app(environ, start_response)

        class CaseResults(WSGIResponse):

            def next(this):
                return getattr(this.iter.next(), self.to)()

            def __next__(this):
                return getattr(next(this.iter), self.to)()
        return CaseResults(res)