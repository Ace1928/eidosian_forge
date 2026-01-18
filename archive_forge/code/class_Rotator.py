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
class Rotator(object):

    def __call__(self, scale):
        r = cherrypy.response
        r.collapse_body()
        r.body = [bytes([(x + scale) % 256 for x in r.body[0]])]