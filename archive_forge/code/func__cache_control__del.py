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
def _cache_control__del(self):
    env = self.environ
    if 'HTTP_CACHE_CONTROL' in env:
        del env['HTTP_CACHE_CONTROL']
    if 'webob._cache_control' in env:
        del env['webob._cache_control']