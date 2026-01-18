import cherrypy
from cherrypy.test import helper
class OurNotClosableIterator(OurIterator):

    def close(self, somearg):
        self.decrement()