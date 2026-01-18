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
class EmptyResponse(object):
    """
    An empty WSGI response.

    An iterator that immediately stops. Optionally provides a close
    method to close an underlying ``app_iter`` it replaces.
    """

    def __init__(self, app_iter=None):
        if app_iter is not None and hasattr(app_iter, 'close'):
            self.close = app_iter.close

    def __iter__(self):
        return self

    def __len__(self):
        return 0

    def next(self):
        raise StopIteration()
    __next__ = next