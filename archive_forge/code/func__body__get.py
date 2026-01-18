import re
import struct
import zlib
from base64 import b64encode
from datetime import datetime, timedelta
from hashlib import md5
from webob.byterange import ContentRange
from webob.cachecontrol import CacheControl, serialize_cache_control
from webob.compat import (
from webob.cookies import Cookie, make_cookie
from webob.datetime_utils import (
from webob.descriptors import (
from webob.headers import ResponseHeaders
from webob.request import BaseRequest
from webob.util import status_generic_reasons, status_reasons, warn_deprecation
def _body__get(self):
    """
        The body of the response, as a :class:`bytes`.  This will read in
        the entire app_iter if necessary.
        """
    app_iter = self._app_iter
    if isinstance(app_iter, list) and len(app_iter) == 1:
        return app_iter[0]
    if app_iter is None:
        raise AttributeError('No body has been set')
    try:
        body = b''.join(app_iter)
    finally:
        iter_close(app_iter)
    if isinstance(body, text_type):
        raise _error_unicode_in_app_iter(app_iter, body)
    self._app_iter = [body]
    if len(body) == 0:
        pass
    elif self.content_length is None:
        self.content_length = len(body)
    elif self.content_length != len(body):
        raise AssertionError('Content-Length is different from actual app_iter length (%r!=%r)' % (self.content_length, len(body)))
    return body