import cherrypy
import io
import logging
import os
import re
import sys
from more_itertools import always_iterable
import cherrypy
from cherrypy._cperror import format_exc, bare_error
from cherrypy.lib import httputil
class _ReadOnlyRequest:
    expose = ('read', 'readline', 'readlines')

    def __init__(self, req):
        for method in self.expose:
            self.__dict__[method] = getattr(req, method)