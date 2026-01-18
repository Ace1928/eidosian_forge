import binascii
import io
import os
import re
import sys
import tempfile
import mimetypes
import warnings
from webob.acceptparse import (
from webob.cachecontrol import (
from webob.compat import (
from webob.cookies import RequestCookies
from webob.descriptors import (
from webob.etag import (
from webob.headers import EnvironHeaders
from webob.multidict import (
def _urlargs__set(self, value):
    environ = self.environ
    if 'paste.urlvars' in environ:
        routing_args = (value, environ.pop('paste.urlvars'))
    elif 'wsgiorg.routing_args' in environ:
        routing_args = (value, environ['wsgiorg.routing_args'][1])
    else:
        routing_args = (value, {})
    environ['wsgiorg.routing_args'] = routing_args