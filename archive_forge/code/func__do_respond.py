import sys
import time
import collections
import operator
from http.cookies import SimpleCookie, CookieError
import uuid
from more_itertools import consume
import cherrypy
from cherrypy._cpcompat import ntob
from cherrypy import _cpreqbody
from cherrypy._cperror import format_exc, bare_error
from cherrypy.lib import httputil, reprconf, encoding
def _do_respond(self, path_info):
    response = cherrypy.serving.response
    if self.app is None:
        raise cherrypy.NotFound()
    self.hooks = self.__class__.hooks.copy()
    self.toolmaps = {}
    self.stage = 'process_headers'
    self.process_headers()
    self.stage = 'get_resource'
    self.get_resource(path_info)
    self.body = _cpreqbody.RequestBody(self.rfile, self.headers, request_params=self.params)
    self.namespaces(self.config)
    self.stage = 'on_start_resource'
    self.hooks.run('on_start_resource')
    self.stage = 'process_query_string'
    self.process_query_string()
    if self.process_request_body:
        if self.method not in self.methods_with_bodies:
            self.process_request_body = False
    self.stage = 'before_request_body'
    self.hooks.run('before_request_body')
    if self.process_request_body:
        self.body.process()
    self.stage = 'before_handler'
    self.hooks.run('before_handler')
    if self.handler:
        self.stage = 'handler'
        response.body = self.handler()
    self.stage = 'before_finalize'
    self.hooks.run('before_finalize')
    response.finalize()