import datetime
from itertools import count
import os
import threading
import time
import urllib.parse
import pytest
import cherrypy
from cherrypy.lib import httputil
from cherrypy.test import helper
@cherrypy.config(**{'tools.expires.on': True, 'tools.expires.secs': 60, 'tools.staticdir.on': True, 'tools.staticdir.dir': 'static', 'tools.staticdir.root': curdir})
class UnCached(object):

    @cherrypy.expose
    @cherrypy.config(**{'tools.expires.secs': 0})
    def force(self):
        cherrypy.response.headers['Etag'] = 'bibbitybobbityboo'
        self._cp_config['tools.expires.force'] = True
        self._cp_config['tools.expires.secs'] = 0
        return 'being forceful'

    @cherrypy.expose
    def dynamic(self):
        cherrypy.response.headers['Etag'] = 'bibbitybobbityboo'
        cherrypy.response.headers['Cache-Control'] = 'private'
        return 'D-d-d-dynamic!'

    @cherrypy.expose
    def cacheable(self):
        cherrypy.response.headers['Etag'] = 'bibbitybobbityboo'
        return "Hi, I'm cacheable."

    @cherrypy.expose
    @cherrypy.config(**{'tools.expires.secs': 86400})
    def specific(self):
        cherrypy.response.headers['Etag'] = 'need_this_to_make_me_cacheable'
        return 'I am being specific'

    class Foo(object):
        pass

    @cherrypy.expose
    @cherrypy.config(**{'tools.expires.secs': Foo()})
    def wrongtype(self):
        cherrypy.response.headers['Etag'] = 'need_this_to_make_me_cacheable'
        return 'Woops'