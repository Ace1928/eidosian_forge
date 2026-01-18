import sys
import cherrypy
from cherrypy._cpcompat import ntob
from cherrypy.test import helper
class WSGIResponse(object):

    def __init__(self, appresults):
        self.appresults = appresults
        self.iter = iter(appresults)

    def __iter__(self):
        return self
    if sys.version_info >= (3, 0):

        def __next__(self):
            return next(self.iter)
    else:

        def next(self):
            return self.iter.next()

    def close(self):
        if hasattr(self.appresults, 'close'):
            self.appresults.close()