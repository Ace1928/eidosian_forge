import cherrypy
from cherrypy.test import helper
class OurUnclosableIterator(OurIterator):
    close = 'close'