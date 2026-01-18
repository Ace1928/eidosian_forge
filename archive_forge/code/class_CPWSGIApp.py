import sys as _sys
import io
import cherrypy as _cherrypy
from cherrypy._cpcompat import ntou
from cherrypy import _cperror
from cherrypy.lib import httputil
from cherrypy.lib import is_closable_iterator
class CPWSGIApp(object):
    """A WSGI application object for a CherryPy Application."""
    pipeline = [('ExceptionTrapper', ExceptionTrapper), ('InternalRedirector', InternalRedirector)]
    "A list of (name, wsgiapp) pairs. Each 'wsgiapp' MUST be a\n    constructor that takes an initial, positional 'nextapp' argument,\n    plus optional keyword arguments, and returns a WSGI application\n    (that takes environ and start_response arguments). The 'name' can\n    be any you choose, and will correspond to keys in self.config."
    head = None
    "Rather than nest all apps in the pipeline on each call, it's only\n    done the first time, and the result is memoized into self.head. Set\n    this to None again if you change self.pipeline after calling self."
    config = {}
    'A dict whose keys match names listed in the pipeline. Each\n    value is a further dict which will be passed to the corresponding\n    named WSGI callable (from the pipeline) as keyword arguments.'
    response_class = AppResponse
    'The class to instantiate and return as the next app in the WSGI chain.\n    '

    def __init__(self, cpapp, pipeline=None):
        self.cpapp = cpapp
        self.pipeline = self.pipeline[:]
        if pipeline:
            self.pipeline.extend(pipeline)
        self.config = self.config.copy()

    def tail(self, environ, start_response):
        """WSGI application callable for the actual CherryPy application.

        You probably shouldn't call this; call self.__call__ instead,
        so that any WSGI middleware in self.pipeline can run first.
        """
        return self.response_class(environ, start_response, self.cpapp)

    def __call__(self, environ, start_response):
        head = self.head
        if head is None:
            head = self.tail
            for name, callable in self.pipeline[::-1]:
                conf = self.config.get(name, {})
                head = callable(head, **conf)
            self.head = head
        return head(environ, start_response)

    def namespace_handler(self, k, v):
        """Config handler for the 'wsgi' namespace."""
        if k == 'pipeline':
            self.pipeline.extend(v)
        elif k == 'response_class':
            self.response_class = v
        else:
            name, arg = k.split('.', 1)
            bucket = self.config.setdefault(name, {})
            bucket[arg] = v