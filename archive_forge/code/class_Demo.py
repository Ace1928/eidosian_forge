import gzip
import io
import sys
import time
import types
import unittest
import operator
from http.client import IncompleteRead
import cherrypy
from cherrypy import tools
from cherrypy._cpcompat import ntou
from cherrypy.test import helper, _test_decorators
@cherrypy.config(**{'tools.nadsat.on': True})
class Demo(Test):

    def index(self, id=None):
        return 'A good piece of cherry pie'

    def ended(self, id):
        return repr(tools.nadsat.ended[id])

    def err(self, id=None):
        raise ValueError()

    def errinstream(self, id=None):
        yield 'nonconfidential'
        raise ValueError()
        yield 'confidential'

    def restricted(self):
        return 'Welcome!'
    restricted = myauthtools.check_access()(restricted)
    userid = restricted

    def err_in_onstart(self):
        return 'success!'

    @cherrypy.config(**{'response.stream': True})
    def stream(self, id=None):
        for x in range(100000000):
            yield str(x)