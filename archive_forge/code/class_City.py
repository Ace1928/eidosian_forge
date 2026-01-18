import os
import importlib
import pytest
import cherrypy
from cherrypy.test import helper
class City:

    def __init__(self, name):
        self.name = name
        self.population = 10000

    @cherrypy.config(**{'tools.response_headers.on': True, 'tools.response_headers.headers': [('Content-Language', 'en-GB')]})
    def index(self, **kwargs):
        return 'Welcome to %s, pop. %s' % (self.name, self.population)

    def update(self, **kwargs):
        self.population = kwargs['pop']
        return 'OK'