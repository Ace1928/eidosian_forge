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
def _host__get(self):
    """Host name provided in HTTP_HOST, with fall-back to SERVER_NAME"""
    if 'HTTP_HOST' in self.environ:
        return self.environ['HTTP_HOST']
    else:
        return '%(SERVER_NAME)s:%(SERVER_PORT)s' % self.environ