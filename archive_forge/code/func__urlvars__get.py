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
def _urlvars__get(self):
    """
        Return any *named* variables matched in the URL.

        Takes values from ``environ['wsgiorg.routing_args']``.
        Systems like ``routes`` set this value.
        """
    if 'paste.urlvars' in self.environ:
        return self.environ['paste.urlvars']
    elif 'wsgiorg.routing_args' in self.environ:
        return self.environ['wsgiorg.routing_args'][1]
    else:
        result = {}
        self.environ['wsgiorg.routing_args'] = ((), result)
        return result