import cherrypy
from cherrypy.test import helper
class OurGenerator(IteratorBase):

    def __iter__(self):
        self.incr()
        try:
            for i in range(1024):
                yield self.datachunk
        finally:
            self.decr()