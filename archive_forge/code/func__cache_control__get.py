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
def _cache_control__get(self):
    """
        Get/set/modify the Cache-Control header (`HTTP spec section 14.9
        <http://www.w3.org/Protocols/rfc2616/rfc2616-sec14.html#sec14.9>`_)
        """
    env = self.environ
    value = env.get('HTTP_CACHE_CONTROL', '')
    cache_header, cache_obj = env.get('webob._cache_control', (None, None))
    if cache_obj is not None and cache_header == value:
        return cache_obj
    cache_obj = CacheControl.parse(value, updates_to=self._update_cache_control, type='request')
    env['webob._cache_control'] = (value, cache_obj)
    return cache_obj