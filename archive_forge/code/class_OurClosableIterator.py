import cherrypy
from cherrypy.test import helper
class OurClosableIterator(OurIterator):

    def close(self):
        self.decrement()