import io
import contextlib
import urllib.parse
from sys import exc_info as _exc_info
from traceback import format_exception as _format_exception
from xml.sax import saxutils
import html
from more_itertools import always_iterable
import cherrypy
from cherrypy._cpcompat import ntob
from cherrypy._cpcompat import tonative
from cherrypy._helper import classproperty
from cherrypy.lib import httputil as _httputil
def _be_ie_unfriendly(status):
    response = cherrypy.serving.response
    s = _ie_friendly_error_sizes.get(status, 0)
    if s:
        s += 1
        content = response.collapse_body()
        content_length = len(content)
        if content_length and content_length < s:
            content = content + b' ' * (s - content_length)
        response.body = content
        response.headers['Content-Length'] = str(len(content))