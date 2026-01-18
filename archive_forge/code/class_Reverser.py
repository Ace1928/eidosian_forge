import sys
import cherrypy
from cherrypy._cpcompat import ntob
from cherrypy.test import helper
class Reverser(WSGIResponse):
    if sys.version_info >= (3, 0):

        def __next__(this):
            line = list(next(this.iter))
            line.reverse()
            return bytes(line)
    else:

        def next(this):
            line = list(this.iter.next())
            line.reverse()
            return ''.join(line)