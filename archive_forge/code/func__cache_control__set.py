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
def _cache_control__set(self, value):
    env = self.environ
    value = value or ''
    if isinstance(value, dict):
        value = CacheControl(value, type='request')
    if isinstance(value, CacheControl):
        str_value = str(value)
        env['HTTP_CACHE_CONTROL'] = str_value
        env['webob._cache_control'] = (str_value, value)
    else:
        env['HTTP_CACHE_CONTROL'] = str(value)
        env['webob._cache_control'] = (None, None)