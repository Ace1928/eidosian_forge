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
def _urlargs__del(self):
    if 'wsgiorg.routing_args' in self.environ:
        if not self.environ['wsgiorg.routing_args'][1]:
            del self.environ['wsgiorg.routing_args']
        else:
            self.environ['wsgiorg.routing_args'] = ((), self.environ['wsgiorg.routing_args'][1])